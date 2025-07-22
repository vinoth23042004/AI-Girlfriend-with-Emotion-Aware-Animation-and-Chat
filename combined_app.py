import os
import threading
import io
import base64
import time
import asyncio
import pygame
import uuid
from io import BytesIO
from PIL import Image
import edge_tts
import numpy as np
import speech_recognition as sr
import sounddevice as sd
import soundfile as sf
from flask import Flask, render_template, request, jsonify, session, Response, send_file
from flask_socketio import SocketIO
from dotenv import load_dotenv
import google.generativeai as genai
import torch
from diffusers import StableDiffusionPipeline

# Import animation modules
# Note: You'll need these modules from your original project
import animate_girl
import chat_with_girl 
import emotion_detector

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Try different async modes or fallback
try:
    # First try with eventlet
    socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")
except ValueError:
    try:
        # Then try with gevent
        socketio = SocketIO(app, async_mode='gevent', cors_allowed_origins="*")
    except ValueError:
        try:
            # Finally try with threading
            socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")
        except ValueError:
            # If all fail, use basic Flask without SocketIO for now
            print("Warning: SocketIO failed to initialize. Running without real-time features.")
            socketio = None

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Configure the Gemini API
genai.configure(api_key=api_key)

# Global variables for animation
frame_buffer = None
frame_lock = threading.Lock()
running = True
VOICE = "en-US-JennyNeural"
stop_audio_event = threading.Event()
emotion_check_active = True
current_emotion = "Neutral"  # Default emotion

# Initialize pygame for audio and animation
pygame.init()
pygame.mixer.init()

# Initialize session variables
@app.before_request
def before_request():
    if 'messages' not in session:
        session['messages'] = []
    if 'voice_messages' not in session:  # Separate history for voice chat
        session['voice_messages'] = []
    if 'generate_images' not in session:
        session['generate_images'] = True
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())

# Initialize the StableDiffusion model
def get_stable_diffusion_model():
    model_path = "D:/Projects/Project 2/Project_Dating/counterfeitV30_v30.safetensors"
    pipe = StableDiffusionPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipe = pipe.to("cuda")
    return pipe

# Global variable for the model to avoid reloading
stable_diffusion_model = None

#
# LEFT SIDE FUNCTIONALITY (Animation and Voice)
#

def capture_pygame_frames():
    """Thread function to continuously capture frames from pygame"""
    global frame_buffer
    
    # Initialize pygame (if not already done)
    if not pygame.get_init():
        pygame.init()
    
    # Get the current display surface
    surface = pygame.display.get_surface()
    
    while running:
        # Get the latest frame from pygame
        with frame_lock:
            # Convert pygame surface to image data
            data = pygame.image.tostring(surface, 'RGB')
            size = surface.get_size()
            
            # Convert to PIL Image
            pil_image = Image.frombytes('RGB', size, data)
            
            # Save to buffer as JPEG
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='JPEG')
            frame_buffer = img_buffer.getvalue()
        
        # Don't hog CPU
        time.sleep(0.03)  # ~30 FPS

@app.route('/video_feed')
def video_feed():
    """Route for streaming the animation as video"""
    def generate():
        global frame_buffer
        while True:
            with frame_lock:
                if frame_buffer:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + 
                           frame_buffer + b'\r\n')
            time.sleep(0.03)  # Match capture rate

    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def lip_sync_animation(audio_data, sample_rate, timestamps):
    """Synchronize lip movement with speech using timestamps."""
    animate_girl.set_speaking(True)
    
    for start, end in timestamps:
        if stop_audio_event.is_set():
            break
        animate_girl.draw_mouth_open()
        time.sleep(end - start)  # Match lip movement to phoneme timing
        animate_girl.draw_mouth_closed()
    
    animate_girl.set_speaking(False)

def generate_timestamps(text, duration, pauses=None):
    """Estimate timestamps for word boundaries while considering pauses."""
    if pauses is None:
        pauses = {}
        
    words = text.split()
    num_words = len(words)
    avg_word_time = duration / max(num_words, 1)
    timestamps = []
    current_time = 0
    
    for i, word in enumerate(words):
        start_time = current_time
        end_time = start_time + avg_word_time
        timestamps.append((start_time, end_time))
        current_time = end_time
        
        if i in pauses:
            current_time += pauses[i]  # Add pause duration at the right positions
    
    return timestamps

async def generate_speech(text):
    """Generate speech audio using Edge TTS"""
    # Create a temporary file for the audio
    temp_file = "temp_speech.mp3"
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(temp_file)
    
    # Read the file into memory and return it
    with open(temp_file, "rb") as f:
        audio_data = io.BytesIO(f.read())
    
    # Clean up the temporary file
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    audio_data.seek(0)
    return audio_data

async def stream_speak(text):
    """Stream speech while synchronizing lip movement with spoken words."""
    communicate = edge_tts.Communicate(text, VOICE)
    
    chunks = []
    pauses = {}  # Store detected pauses
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
        elif chunk["type"] == "bookmark":
            pauses[len(chunks)] = float(chunk["value"])  # Store pause durations
    
    if chunks:
        combined_audio = b''.join(chunks)
        with sf.SoundFile(io.BytesIO(combined_audio)) as audio_file:
            audio_data = audio_file.read(dtype='float32')
            sample_rate = audio_file.samplerate
        
        duration = len(audio_data) / sample_rate
        timestamps = generate_timestamps(text, duration, pauses)
        
        stop_audio_event.clear()
        animation_thread = threading.Thread(target=lip_sync_animation, args=(audio_data, sample_rate, timestamps))
        animation_thread.start()
        
        sd.play(audio_data, sample_rate)
        sd.wait()
        
        stop_audio_event.set()
        animation_thread.join()
        animate_girl.set_speaking(False)

def listen():
    """Listen to user speech and return recognized text."""
    global current_emotion
    
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Speak now...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            print(f"🗣 You said: {text}")
            
            # Update emotion based on what was said
            current_emotion = emotion_detector.detect_emotion()
            
            return text.lower()
        except (sr.UnknownValueError, sr.RequestError, sr.WaitTimeoutError):
            print("❌ Could not understand, please try again.")
            return None

def periodic_emotion_check():
    """Thread function to periodically check user's emotion"""
    global emotion_check_active, current_emotion
    
    while emotion_check_active:
        time.sleep(10)  # Check every 10 seconds
        
        # Only perform checks if the system is running
        if not running:
            break
            
        detected_emotion = emotion_detector.detect_emotion()
        current_emotion = detected_emotion
        
        # Broadcast the updated emotion to all clients
        if socketio:
            socketio.emit('emotion_update', {'emotion': current_emotion})
        
        # Automatic empathetic response for negative emotions
        if detected_emotion in ["sad", "angry", "disgust"]:
            caring_response = f"You seem {detected_emotion}. Is everything okay? Want to talk about it?"
            print(f"🤖 AI Girl: {caring_response}")
            
            # Generate audio response
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(stream_speak(caring_response))
                
                # Get user response via microphone
                user_reply = listen()
                if user_reply:
                    ai_response = chat_with_girl.chat_with_ai(user_reply)
                    print(f"💬 AI Girl: {ai_response}")
                    loop.run_until_complete(stream_speak(ai_response))
            except Exception as e:
                print(f"Error in emotion response: {e}")
            finally:
                loop.close()

@app.route('/get_current_emotion')
def get_current_emotion():
    """API endpoint to get the current detected emotion"""
    global current_emotion
    return jsonify({'emotion': current_emotion})

@app.route('/process_voice', methods=['POST'])
def process_voice():
    """Process voice input from the left side only"""
    message = request.json.get('message', '')
    
    # Store in separate voice message history
    voice_messages = session.get('voice_messages', [])
    voice_messages.append({"role": "user", "content": message})
    
    # Get AI response specifically for voice interaction
    response = chat_with_girl.chat_with_ai(message)
    
    # Add AI response to voice message history
    voice_messages.append({"role": "assistant", "content": response})
    session['voice_messages'] = voice_messages
    
    # Generate audio and get duration for lip sync
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        audio_data = loop.run_until_complete(generate_speech(response))
        
        # Get actual audio duration
        audio_file = io.BytesIO(audio_data.getvalue())
        sound = pygame.mixer.Sound(audio_file)
        duration = sound.get_length()
        
        # Detect emotion from response
        emotion = detect_emotion_from_response(response)
        
        # Update global emotion state
        global current_emotion
        current_emotion = emotion
        
        # Broadcast the updated emotion
        if socketio:
            socketio.emit('emotion_update', {'emotion': emotion})
        
        return jsonify({
            'response': response,
            'audio_available': True,
            'duration': duration,
            'emotion': emotion
        })
    except Exception as e:
        print(f"Error generating audio for voice: {e}")
        return jsonify({'response': response, 'audio_available': False})
    finally:
        loop.close()

@app.route('/get_audio/<message>')
def get_audio(message):
    """Generate and return audio for the given message"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        audio_data = loop.run_until_complete(generate_speech(message))
        
        return send_file(
            audio_data,
            mimetype='audio/mpeg',
            as_attachment=True,
            download_name='speech.mp3'
        )
    except Exception as e:
        print(f"Error generating audio: {e}")
        return "Error generating audio", 500
    finally:
        loop.close()

@app.route('/start_listening')
def start_listening():
    """API endpoint to start listening for voice input"""
    user_text = listen()
    if user_text:
        return jsonify({'text': user_text, 'success': True, 'emotion': current_emotion})
    else:
        return jsonify({'success': False, 'message': 'Could not understand speech', 'emotion': current_emotion})

@app.route('/start_speaking/<float:duration>')
def start_speaking(duration):
    """Start speaking animation for exact duration"""
    animate_girl.set_speaking(True)
    threading.Timer(duration, lambda: animate_girl.set_speaking(False)).start()
    return jsonify(success=True)

@app.route('/stop_speaking')
def stop_speaking():
    """Stop speaking animation"""
    animate_girl.set_speaking(False)
    return jsonify(success=True)

@app.route('/voice_input')
def voice_input():
    """API endpoint to receive voice input from the user"""
    global current_emotion
    
    user_text = listen()
    if user_text:
        return jsonify({'text': user_text, 'success': True, 'emotion': current_emotion})
    else:
        return jsonify({'success': False, 'message': 'Could not understand speech', 'emotion': current_emotion})

#
# RIGHT SIDE FUNCTIONALITY (Text Chat with Image Generation)
#

# Function to detect emotion from response
def detect_emotion_from_response(text):
    """
    Detect the emotion conveyed by the AI's response, considering the overall tone
    rather than just keyword presence.
    Returns one of: happy, sad, angry, surprised, neutral, caring, loving
    """
    text = text.lower()
    
    # Check for emotional patterns in the response
    # For supportive/caring responses when user is sad
    if any(phrase in text for phrase in ["sorry you're feeling", "here for you", "virtual hug", "tell me about it", 
                                        "feel better", "cheer you up", "i'm here", "i care", "oh no", "hope you"]):
        return "caring"
        
    # For loving/affectionate responses
    if any(phrase in text for phrase in ["love you", "miss you", "thinking of you", "sweetheart", "darling", 
                                        "cuddle", "kiss", "embrace", "adore"]):
        return "loving"
    
    # Standard emotion detection for other cases
    happy_keywords = ['happy', 'glad', 'joy', 'excited', 'smile', 'yay', 'haha', '😊', '😄', '❤️']
    sad_keywords = ['sad', 'sorry', 'miss', 'upset', 'cry', 'tears', 'disappointed', '😢', '😭', '💔']
    angry_keywords = ['angry', 'mad', 'frustrat', 'annoyed', 'upset', 'grr', '😠', '😡']
    surprised_keywords = ['wow', 'omg', 'oh my', 'surprised', 'what!', 'really?', 'amazing', '😲', '😮', '😱']
    
    # Count occurrences of emotion keywords
    happy_count = sum(1 for word in happy_keywords if word in text)
    sad_count = sum(1 for word in sad_keywords if word in text)
    angry_count = sum(1 for word in angry_keywords if word in text)
    surprised_count = sum(1 for word in surprised_keywords if word in text)
    
    # Find the dominant emotion
    emotion_counts = {
        'happy': happy_count,
        'sad': sad_count,
        'angry': angry_count,
        'surprised': surprised_count
    }
    
    max_emotion = max(emotion_counts, key=emotion_counts.get)
    
    # Return neutral if no clear emotion is detected
    if emotion_counts[max_emotion] == 0:
        return 'neutral'
    
    return max_emotion

# Function to generate images based on emotion
def generate_image(emotion):
    global stable_diffusion_model
    
    # Initialize the model if not already loaded
    if stable_diffusion_model is None:
        stable_diffusion_model = get_stable_diffusion_model()
    
    # Direct, explicit prompts for better anime girl expressions
    emotion_prompts = {
        "happy": "Create a beautiful anime girl with a joyful expression, sparkling eyes, wide smile, surrounded by bright, warm colors like yellow and orange, with a joyful, uplifting atmosphere, highly detailed and vibrant.",
        "sad": "Create an anime girl with a sorrowful expression, gentle tears on her cheeks, eyes looking down, soft, muted lighting in shades of blue and gray, evoking a melancholic, reflective mood, high detail.",
        "angry": "Create an anime girl with an angry expression, furrowed brows, clenched fists, and a piercing gaze, a dark red and black background with a fiery atmosphere, detailed and intense.",
        "surprised": "Create an anime girl with wide, astonished eyes, mouth slightly open, and raised eyebrows, a background with light, pastel colors creating a bright, energetic feel, very detailed.",
        "neutral": "Create an anime girl with a calm, serene expression, soft eyes, gentle smile, ambient lighting with light pastel tones, creating a peaceful, quiet atmosphere, highly detailed.",
        "caring": "Create an anime girl with a compassionate expression, warm, soft smile, comforting eyes, her hands in a gentle gesture like a hug or holding a flower, soft pink and lavender lighting in the background, detailed.",
        "loving": "Create an anime girl with a loving expression, soft blush on her cheeks, tender smile, eyes filled with affection, soft warm lighting like a sunset, romantic background, detailed.",
        "possessive": "Create an anime girl with a possessive expression, eyes locked onto the viewer with intense focus, a slight smirk or jealous glance, close body language, deep red and purple hues, dramatic lighting creating a bold, captivating atmosphere, highly detailed.",
        "shy": "Create an anime girl with a shy expression, slightly averted eyes, blushing cheeks, hands clasped nervously, a soft pastel background with gentle lighting in pink and sky blue, evoking a sweet and innocent vibe, very detailed.",
        "excited": "Create an anime girl with an excited expression, sparkling wide eyes, open-mouthed smile, hands raised in joy or surprise, vibrant colorful background with confetti or sparkles, energetic and cheerful mood, very detailed.",
        "confident": "Create an anime girl with a confident expression, hands on hips or striking a bold pose, intense, determined eyes, with a powerful atmosphere created by dynamic lighting and a background in bold colors like crimson and gold, highly detailed and heroic.",
        "worried": "Create an anime girl with a worried expression, eyebrows slightly furrowed, lips parted as if about to speak, hands held close to chest, a soft cloudy background in muted gray and blue tones, conveying concern, detailed and expressive."
    }
    
    # If the user is sad but AI response is positive, use caring expression
    prompt = emotion_prompts.get(emotion, emotion_prompts["neutral"])
    
    # Generate image with more inference steps for better quality
    with torch.no_grad():
        image = stable_diffusion_model(prompt, num_inference_steps=35).images[0]
    
    # Convert PIL image to base64 for sending to frontend
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

# Function to get response from Gemini API using available model
def get_gemini_response(input_text, chat_history):
    try:
        # Use one of the available models from your list
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        
        # Prepare conversation context
        system_prompt = "You are an AI girlfriend. Respond in a sweet, caring manner. Keep responses short and personal. Show emotions through your text."
        
        # Format the conversation for the API
        messages = [
            {"role": "user", "parts": [system_prompt]},
            {"role": "model", "parts": ["I understand! I'll be your sweet AI girlfriend, responding with care and emotion."]}
        ]
        
        # Add relevant history
        recent_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
        for msg in recent_history:
            if msg["role"] == "user":
                messages.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                messages.append({"role": "model", "parts": [msg["content"]]})
        
        # Add current user message
        messages.append({"role": "user", "parts": [input_text]})
        
        # Generate content
        response = model.generate_content(messages)
        
        return response.text
        
    except Exception as e:
        print(f"Error with Gemini API: {str(e)}")
        
        # Try with alternate model if first fails
        try:
            # Try with a different model from your list as backup
            alternate_model = genai.GenerativeModel("models/gemini-1.5-pro")
            
            # Just send a single prompt to keep it simple for backup method
            prompt = f"{system_prompt}\n\nUser message: {input_text}\n\nYour response:"
            response = alternate_model.generate_content(prompt)
            return response.text
            
        except Exception as alt_error:
            print(f"Error with alternate model: {str(alt_error)}")
            
            # Last resort - try with text-only model
            try:
                text_model = genai.GenerativeModel("models/text-bison-001")
                response = text_model.generate_content(
                    f"Act as a sweet AI girlfriend responding to this message: '{input_text}'. Keep your response short and emotional."
                )
                return response.text
            except:
                # Final fallback response
                return "I'm having trouble connecting right now. Could you say that again? (API Error)"

@app.route('/send_message', methods=['POST'])
def send_message():
    user_message = request.json.get('message', '')
    generate_images = request.json.get('generate_images', True)
    
    # Update session state for image generation preference
    session['generate_images'] = generate_images
    
    # Add user message to chat history (RIGHT SIDE ONLY)
    messages = session.get('messages', [])
    messages.append({"role": "user", "content": user_message})
    
    # Get response from Gemini
    ai_response = get_gemini_response(user_message, messages)
    
    response_data = {
        "role": "assistant",
        "content": ai_response,
    }
    
    # Generate image if toggle is on
    if generate_images:
        try:
            emotion = detect_emotion_from_response(ai_response)
            image_data = generate_image(emotion)
            response_data["image"] = image_data
            response_data["emotion"] = emotion
            
            # Update the global emotion state to keep both sides in sync
            global current_emotion
            current_emotion = emotion
            
            # Emit the emotion update to ensure animated character reacts appropriately
            if socketio:
                socketio.emit('emotion_update', {'emotion': current_emotion})
        except Exception as e:
            print(f"Error generating image: {str(e)}")
    
    # Add AI response to chat history (RIGHT SIDE ONLY)
    messages.append(response_data)
    session['messages'] = messages
    
    return jsonify(response_data)

@app.route('/toggle_images', methods=['POST'])
def toggle_images():
    generate_images = request.json.get('generate_images', True)
    session['generate_images'] = generate_images
    return jsonify({"status": "success", "generate_images": generate_images})

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    # Clear both chat histories
    session['messages'] = []
    session['voice_messages'] = []
    return jsonify({"status": "success"})

# 
# SHARED ROUTES
#

@app.route('/')
def index():
    return render_template('index.html', 
                          messages=session.get('messages', []), 
                          generate_images=session.get('generate_images', True))

@app.route('/connect_sides', methods=['POST'])
def connect_sides():
    """Bridge between the text chat and animated character"""
    data = request.json
    message_type = data.get('type')
    content = data.get('content')
    
    if message_type == 'text_to_voice':
        # Pass text message from right side to the left side for voice response
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(stream_speak(content))
            return jsonify({"status": "success"})
        except Exception as e:
            print(f"Error in text to voice: {str(e)}")
            return jsonify({"status": "error", "message": str(e)})
    
    elif message_type == 'voice_to_text':
        # Pass voice input from left side to right side as text message
        return jsonify({"status": "success", "content": content})
    
    return jsonify({"status": "error", "message": "Invalid connection type"})

if __name__ == '__main__':
    # Start the animation thread
    animation_thread = threading.Thread(target=animate_girl.run_animation)
    animation_thread.daemon = True
    animation_thread.start()
    
    # Start the frame capture thread
    capture_thread = threading.Thread(target=capture_pygame_frames)
    capture_thread.daemon = True
    capture_thread.start()
    
    # Start emotion detection thread
    emotion_thread = threading.Thread(target=periodic_emotion_check)
    emotion_thread.daemon = True
    emotion_thread.start()
    
    try:
        # Run the Flask app with Socket.IO if available, otherwise just Flask
        if socketio:
            socketio.run(app, debug=False, host='0.0.0.0', port=5000)
        else:
            app.run(debug=False, host='0.0.0.0', port=5000)
    finally:
        # Clean up
        running = False
        emotion_check_active = False
        pygame.quit()