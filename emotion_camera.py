import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import pygame
import time

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="AI Emotion Music Player", layout="wide")
st.title("MoodSync: Emotion-Based Music Player")
run = st.button('Start Camera')
FRAME_WINDOW = st.image([]) # Placeholder for the video feed

# --- INITIALIZATION ---
# Use @st.cache_resource so the model only loads once
@st.cache_resource
def load_emotion_model():
    return load_model("emotion_model.hdf5")

model = load_emotion_model()
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Initialize pygame mixer (Keep outside the loop)
if 'mixer_init' not in st.session_state:
    pygame.mixer.init()
    st.session_state.mixer_init = True

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Emotion Logic Variables
if 'last_emotion' not in st.session_state:
    st.session_state.last_emotion = None
    st.session_state.emotion_history = []

history_size = 10
music_cooldown = 3
last_music_change = 0

def play_music(emotion):
    pygame.mixer.music.stop()
    try:
        if emotion == "Happy":
            pygame.mixer.music.load("music/happy.mp3")
        elif emotion == "Sad":
            pygame.mixer.music.load("music/sad.mp3")
        elif emotion == "Angry":
            pygame.mixer.music.load("music/angry.mp3")
        else:
            pygame.mixer.music.load("music/neutral.mp3")
        pygame.mixer.music.play()
    except Exception as e:
        st.error(f"Error loading music: {e}")

# --- MAIN LOOP ---
# Use a standard IP cam or 0 for local webcam
cap = cv2.VideoCapture("http://192.168.29.154:8080/video")

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to fetch video. Check IP Camera address.")
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    stable_emotion = st.session_state.last_emotion

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48)) / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        prediction = model.predict(face, verbose=0)
        emotion = emotion_labels[np.argmax(prediction)]

        st.session_state.emotion_history.append(emotion)
        if len(st.session_state.emotion_history) > history_size:
            st.session_state.emotion_history.pop(0)

        stable_emotion = max(set(st.session_state.emotion_history), key=st.session_state.emotion_history.count)

        # Logic for music change
        current_time = time.time()
        if stable_emotion != st.session_state.last_emotion and (current_time - last_music_change > music_cooldown):
            play_music(stable_emotion)
            st.session_state.last_emotion = stable_emotion
            last_music_change = current_time

        # Draw UI on frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, stable_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Convert BGR to RGB for Streamlit display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame_rgb)

else:
    cap.release()
    st.write("Camera is off.")