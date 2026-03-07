import cv2
from deepface import DeepFace
import pygame
import time
import streamlit as st

pygame.mixer.init()

cap = cv2.VideoCapture("http://192.168.0.104:8080/video")

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

last_emotion = None
last_check = 0
music_playing = False


def play_music(emotion):
    global music_playing

    pygame.mixer.music.stop()

    if emotion == "happy":
        pygame.mixer.music.load("music/happy.mp3")

    elif emotion == "sad":
        pygame.mixer.music.load("music/sad.mp3")

    elif emotion == "angry":
        pygame.mixer.music.load("music/angry.mp3")

    else:
        pygame.mixer.music.load("music/neutral.mp3")

    pygame.mixer.music.play()
    music_playing = True


while True:

    ret, frame = cap.read()

    # prevent crash if stream drops
    if not ret or frame is None:
        print("Frame not received")
        continue

    frame = cv2.resize(frame,(640,480))  # faster processing

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:

        if music_playing:
            pygame.mixer.music.stop()
            music_playing = False

        last_emotion = None

        cv2.imshow("Emotion Music AI", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        continue

    for (x, y, w, h) in faces:

        face_img = frame[y:y+h, x:x+w]

        try:

            if time.time() - last_check > 1:

                result = DeepFace.analyze(
                    face_img,
                    actions=['emotion'],
                    enforce_detection=False
                )

                emotions = result[0]['emotion']

                if emotions["angry"] > 8:
                    emotion = "angry"
                else:
                    allowed = {
                        "happy": emotions["happy"],
                        "sad": emotions["sad"],
                        "neutral": emotions["neutral"]
                    }

                    emotion = max(allowed, key=allowed.get)


            # FOR STABALZING THE EMOTIONS
                last_music_change = 0
                music_cooldown = 5

                if emotion != last_emotion and time.time() - last_music_change > music_cooldown:
                    play_music(emotion)
                    last_emotion = emotion
                    last_music_change = time.time()

                last_check = time.time()

        except:
            pass

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        if last_emotion:
            cv2.putText(
                frame,
                last_emotion,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

    cv2.imshow("Emotion Music AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()