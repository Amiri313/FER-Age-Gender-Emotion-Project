"""Run the trained models locally with a webcam.

Place these files in the project root:
    emotion_model.keras
    age_gender_model_finetuned.keras
    haarcascade_frontalface_default.xml

Run:
    python webcam_app.py

Press q to quit.
"""
import os
import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMOTION_MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.keras")
AGE_GENDER_MODEL_PATH = os.path.join(BASE_DIR, "age_gender_model_finetuned.keras")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

for path in (EMOTION_MODEL_PATH, AGE_GENDER_MODEL_PATH, CASCADE_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")

emotion_model = load_model(EMOTION_MODEL_PATH)
age_gender_model = load_model(AGE_GENDER_MODEL_PATH)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
AGE_BRACKETS = ["0-12", "13-19", "20-29", "30-39", "40-49", "50-59", "60+"]
GENDER_LABELS = ["Male", "Female"]


def preprocess_for_emotion(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    return normalized.reshape(1, 48, 48, 1)


def preprocess_for_age_gender(face_bgr):
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (128, 128))
    normalized = resized / 255.0
    return normalized.reshape(1, 128, 128, 3)


def main():
    cap = cv2.VideoCapture(0)
    previous_time = time.time()

    if not cap.isOpened():
        print("Could not open the webcam.")
        return

    window_name = "Facial Emotion, Age & Gender Detection"

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=8, minSize=(100, 100)
        )

        for x, y, w, h in faces:
            face_crop = frame[y:y + h, x:x + w]

            emotion_probs = emotion_model.predict(
                preprocess_for_emotion(face_crop), verbose=0
            )[0]
            emotion_idx = int(np.argmax(emotion_probs))
            emotion_label = EMOTION_CLASSES[emotion_idx]
            emotion_conf = float(emotion_probs[emotion_idx]) * 100

            age_probs, gender_probs = age_gender_model.predict(
                preprocess_for_age_gender(face_crop), verbose=0
            )
            age_idx = int(np.argmax(age_probs[0]))
            age_label = AGE_BRACKETS[age_idx]
            age_conf = float(age_probs[0][age_idx]) * 100

            gender_prob = float(gender_probs[0][0])
            gender_label = GENDER_LABELS[1] if gender_prob > 0.5 else GENDER_LABELS[0]
            gender_conf = (gender_prob if gender_prob > 0.5 else 1 - gender_prob) * 100

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion_label} ({emotion_conf:.0f}%)",
                        (x, max(25, y - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"{age_label} | {gender_label} ({age_conf:.0f}% / {gender_conf:.0f}%)",
                        (x, max(45, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        current_time = time.time()
        fps = 1 / max(current_time - previous_time, 1e-6)
        previous_time = current_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
