# 💖 AI Girlfriend - Emotion-Aware Animated Chatbot

An immersive AI girlfriend web app that combines real-time **emotion detection**, **anime-style lip sync animation**, **voice-to-text conversation**, and **emotion-driven image generation**. This project aims to simulate a virtual girlfriend who responds visually and emotionally to your text or voice input.

---

## 🌟 Demo

### 🎥 Video Demo

[![Watch the demo](https://img.youtube.com/vi/<demo-video-id>/0.jpg)](https://www.youtube.com/watch?v=<demo-video-id>)

> Replace `<demo-video-id>` with your actual YouTube video ID or upload a `.mp4` inside the repo and embed it using HTML.

### 🖼️ Screenshots

| Emotion Response | Voice Chat |
|------------------|------------|
| ![Happy Response](assets/screenshots/happy.png) | ![Voice Chat](assets/screenshots/voice_chat.png) |

---

## 🧠 Tech Stack

| Layer              | Technology |
|-------------------|------------|
| **Frontend**       | HTML5, CSS3, JavaScript, Bootstrap |
| **Backend**        | Flask (Python) |
| **AI Chat Model**  | Google Gemini API (gemini-1.5-flash, pro) |
| **Text-to-Speech** | Microsoft Edge TTS (via `edge-tts`) |
| **Emotion Detection** | DeepFace (OpenCV + Deep Learning) |
| **Voice Input**    | `speech_recognition`, `sounddevice`, `soundfile` |
| **Image Generation** | Stable Diffusion via HuggingFace `diffusers` |
| **Animation Engine** | Pygame (for anime-style mouth and face movements) |
| **Webcam Access**  | OpenCV |
| **Live Interaction** | Flask-SocketIO |

---


---

## 💡 Key Features Explained

### 🎤 Voice Chat with Emotion Recognition

- Speak through the microphone.
- Your emotion is detected via webcam using `DeepFace`.
- The AI girlfriend adapts her response (and voice tone) to your mood.

### 👄 Lip-Synced Anime Animation

- Pygame engine animates a girl with **blinking and speaking** motions.
- During speech, her lips move in sync using timing estimation.

### 💬 Text Chat Interface

- You can also interact through text.
- Gemini API provides short, emotional, and contextual responses.
- Session memory is maintained across interactions.

### 🎨 Emotion-Based Image Generation

- Based on your detected or implied emotion, an **anime image** is generated.
- This is done using a custom Stable Diffusion model.

### 🌈 Emotion Response Types

| Emotion     | Visual Change |
|-------------|----------------|
| Happy       | Bright colors, joyful face |
| Sad         | Blue tone, tears |
| Angry       | Red background, intense eyes |
| Caring      | Soft lighting, hug gesture |
| Loving      | Romantic mood, blushing cheeks |
| Shy         | Blushing, shy eyes |

---

## 🔧 Installation Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/ai-girlfriend
cd ai-girlfriend

