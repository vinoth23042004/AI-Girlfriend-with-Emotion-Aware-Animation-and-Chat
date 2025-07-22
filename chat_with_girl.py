import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

# ✅ Load .env file
load_dotenv("D:\\Projects\\Project 2\\Project_Dating\\.env")

# ✅ Fetch API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ✅ Check if API key is loaded correctly
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY is missing! Check your .env file.")

# ✅ Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# ✅ Use the correct model name for Gemini free tier
MODEL_NAME = "gemini-1.5-flash"  # Fixed: was "gemini-1.5" which doesn't exist

# ✅ File paths
PROMPTS_FILE = "prompts.txt"
HISTORY_FILE = "conversation_history.json"

# ✅ Load predefined prompts
def load_prompts():
    try:
        with open(PROMPTS_FILE, "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        print("⚠️ prompts.txt not found, using default prompt")
        return "You are a sweet AI girlfriend. Respond in a caring, affectionate manner. Keep responses conversational and emotional."

# ✅ Load conversation history
def load_conversation_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as file:
                return json.load(file)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"⚠️ Error loading conversation history: {e}")
            return []
    return []

# ✅ Save conversation history
def save_conversation_history(history):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as file:
            json.dump(history, file, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Error saving conversation history: {e}")

# ✅ Chat function with conversation memory and better error handling
def chat_with_ai(user_input):
    """
    Get AI response while maintaining conversation history.
    """
    try:
        print(f"🗣️ User input: {user_input}")
        
        # Load conversation history
        history = load_conversation_history()
        
        # Prepare context from recent conversations (last 5 exchanges)
        context_messages = []
        recent_history = history[-5:] if len(history) > 5 else history
        
        for msg in recent_history:
            context_messages.append(f"User: {msg['user']}")
            context_messages.append(f"Assistant: {msg['ai']}")
        
        context = "\n".join(context_messages)
        
        # Load system prompts
        prompts = load_prompts()
        
        # Create the full prompt for the AI
        if context:
            full_prompt = f"""System Instructions: {prompts}

Previous conversation:
{context}

Current message:
User: {user_input}

Please respond as the AI girlfriend:"""
        else:
            full_prompt = f"""System Instructions: {prompts}

User: {user_input}

Please respond as the AI girlfriend:"""
        
        print(f"🤖 Sending prompt to Gemini API...")
        
        # Initialize the model
        model = genai.GenerativeModel(MODEL_NAME)
        
        # Generate response with proper configuration
        generation_config = genai.GenerationConfig(
            temperature=0.7,
            top_p=0.8,
            top_k=40,
            max_output_tokens=500,
        )
        
        response = model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        
        # Extract the response text
        if response and hasattr(response, 'text') and response.text:
            ai_response = response.text.strip()
            print(f"✅ AI Response: {ai_response}")
        else:
            print("❌ Empty response from API")
            ai_response = "I'm having trouble thinking of what to say right now. Could you try asking me something else?"
        
        # Save to conversation history
        history.append({"user": user_input, "ai": ai_response})
        save_conversation_history(history)
        
        return ai_response
        
    except Exception as e:
        print(f"❌ Detailed error in AI response: {str(e)}")
        print(f"❌ Error type: {type(e).__name__}")
        
        # Provide more specific error messages based on the error type
        if "API_KEY" in str(e).upper():
            return "There's an issue with my API key. Please check the configuration."
        elif "QUOTA" in str(e).upper() or "LIMIT" in str(e).upper():
            return "I've reached my usage limit for now. Please try again later."
        elif "NETWORK" in str(e).upper() or "CONNECTION" in str(e).upper():
            return "I'm having trouble connecting. Please check your internet connection."
        elif "MODEL" in str(e).upper():
            return "There's an issue with my AI model. Please try again."
        else:
            return f"I'm having some technical difficulties right now. Error: {str(e)[:100]}..."

# ✅ Test function to verify API connectivity
def test_api_connection():
    """Test if the API is working correctly"""
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content("Hello, please respond with 'API test successful'")
        if response and response.text:
            print("✅ API test successful!")
            return True
        else:
            print("❌ API test failed - no response")
            return False
    except Exception as e:
        print(f"❌ API test failed: {str(e)}")
        return False

# Run API test when module is imported
if __name__ == "__main__":
    print("🧪 Testing API connection...")
    test_api_connection()