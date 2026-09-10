"""Streamlit Community Cloud app: facial emotion, age-group, and gender prediction.

Deploy on https://share.streamlit.io — set "Main file path" to
streamlit_app/streamlit_app.py when connecting this repo.

Expected repo layout (this file sits in streamlit_app/, one level below root):
    <repo root>/
    ├── haarcascade_frontalface_default.xml
    ├── models/
    │   ├── emotion_model.keras
    │   └── age_gender_model_finetuned.keras
    └── streamlit_app/
        ├── streamlit_app.py   <- this file
        └── requirements.txt
"""
import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

EMOTION_MODEL_PATH = os.path.join(MODELS_DIR, "emotion_model.keras")
AGE_GENDER_MODEL_PATH = os.path.join(MODELS_DIR, "age_gender_model_finetuned.keras")
CASCADE_PATH = os.path.join(PROJECT_ROOT, "haarcascade_frontalface_default.xml")

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
AGE_BRACKETS = ["0-12", "13-19", "20-29", "30-39", "40-49", "50-59", "60+"]
GENDER_LABELS = ["Male", "Female"]
CONFIDENCE_WARN_THRESHOLD = 50.0


@st.cache_resource(show_spinner="Loading models…")
def load_models():
    emotion_model = load_model(EMOTION_MODEL_PATH)
    age_gender_model = load_model(AGE_GENDER_MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    return emotion_model, age_gender_model, face_cascade


def preprocess_for_emotion(face_rgb):
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    return normalized.reshape(1, 48, 48, 1)


def preprocess_for_age_gender(face_rgb):
    resized = cv2.resize(face_rgb, (128, 128))
    normalized = resized / 255.0
    return normalized.reshape(1, 128, 128, 3)


def predict_face(face_rgb, emotion_model, age_gender_model):
    emotion_probs = emotion_model.predict(preprocess_for_emotion(face_rgb), verbose=0)[0]
    emotion_idx = int(np.argmax(emotion_probs))

    age_probs, gender_probs = age_gender_model.predict(
        preprocess_for_age_gender(face_rgb), verbose=0
    )
    age_idx = int(np.argmax(age_probs[0]))
    gender_prob = float(gender_probs[0][0])
    gender_idx = 1 if gender_prob > 0.5 else 0

    return {
        "emotion_label": EMOTION_CLASSES[emotion_idx],
        "emotion_conf": float(emotion_probs[emotion_idx]) * 100,
        "age_label": AGE_BRACKETS[age_idx],
        "age_conf": float(age_probs[0][age_idx]) * 100,
        "gender_label": GENDER_LABELS[gender_idx],
        "gender_conf": (gender_prob if gender_prob > 0.5 else 1 - gender_prob) * 100,
    }


def draw_predictions(image_rgb, x, y, w, h, result):
    min_conf = min(result["emotion_conf"], result["age_conf"], result["gender_conf"])
    color = (0, 255, 0) if min_conf >= CONFIDENCE_WARN_THRESHOLD else (255, 200, 0)

    cv2.rectangle(image_rgb, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        image_rgb, f"{result['emotion_label']} ({result['emotion_conf']:.0f}%)",
        (x, max(25, y - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
    )
    cv2.putText(
        image_rgb,
        f"{result['age_label']} | {result['gender_label']} "
        f"({result['age_conf']:.0f}% / {result['gender_conf']:.0f}%)",
        (x, max(45, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
    )


def analyze(image_rgb, emotion_model, age_gender_model, face_cascade):
    image_rgb = image_rgb.copy()
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=8, minSize=(100, 100)
    )

    for x, y, w, h in faces:
        face_crop = image_rgb[y:y + h, x:x + w]
        result = predict_face(face_crop, emotion_model, age_gender_model)
        draw_predictions(image_rgb, x, y, w, h, result)

    return image_rgb, len(faces)


def main():
    st.set_page_config(page_title="Emotion, Age & Gender Prediction", page_icon="🙂")
    st.title("Facial Emotion, Age Group & Gender Prediction")
    st.caption(
        "Validation accuracy — emotion: 59.4% · age group: 58.1% · gender: 87.2%. "
        "Full breakdown in the project report on GitHub."
    )
    st.info(
        "Predictions are probabilistic model outputs and can be wrong. Not a reliable "
        "judgment of identity, personality, or actual demographic characteristics.",
        icon="ℹ️",
    )

    emotion_model, age_gender_model, face_cascade = load_models()

    tab_camera, tab_upload = st.tabs(["Camera snapshot", "Upload a photo"])

    source_image = None
    with tab_camera:
        camera_file = st.camera_input("Take a snapshot")
        if camera_file is not None:
            source_image = Image.open(camera_file).convert("RGB")

    with tab_upload:
        uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            source_image = Image.open(uploaded_file).convert("RGB")

    if source_image is None:
        st.stop()

    image_rgb = np.array(source_image)
    annotated, face_count = analyze(image_rgb, emotion_model, age_gender_model, face_cascade)

    if face_count == 0:
        st.warning("No face detected in this image.")
    else:
        st.success(f"Detected {face_count} face(s).")

    st.image(annotated, use_column_width=True)


if __name__ == "__main__":
    main()
