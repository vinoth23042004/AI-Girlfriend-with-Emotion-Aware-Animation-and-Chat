import cv2
from deepface import DeepFace

def detect_emotion():
    """
    Capture a frame from the webcam and detect dominant emotion.
    Returns the emotion if face is detected, else None.
    """
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Failed to capture frame.")
        return None

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion']
        print(f"😊 Detected emotion: {dominant_emotion}")
        return dominant_emotion
    except Exception as e:
        print(f"❌ Emotion detection failed: {e}")
        return None
