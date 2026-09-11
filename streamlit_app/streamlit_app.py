"""Streamlit web app for facial emotion, age-group, and gender prediction."""
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

st.set_page_config(page_title="Facial Analysis | Deep Learning", page_icon="🙂", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px;}
.hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(135deg,#0f172a,#1e3a8a); color: white; margin-bottom: 1rem;}
.hero h1 {margin: 0; font-size: 2.1rem;}
.hero p {margin: .4rem 0 0; color: #dbeafe;}
.metric-card {padding: 1rem; border-radius: 14px; border: 1px solid #e2e8f0; background: #f8fafc; margin-bottom: .7rem;}
.metric-title {font-size: .82rem; color: #64748b; text-transform: uppercase; letter-spacing: .04em;}
.metric-value {font-size: 1.35rem; font-weight: 700; margin-top: .2rem;}
.small-note {color:#64748b;font-size:.85rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading trained models…")
def load_models():
    emotion_model = load_model(EMOTION_MODEL_PATH)
    age_gender_model = load_model(AGE_GENDER_MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    return emotion_model, age_gender_model, face_cascade


def preprocess_for_emotion(face_rgb):
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    return (cv2.resize(gray, (48, 48)) / 255.0).reshape(1, 48, 48, 1)


def preprocess_for_age_gender(face_rgb):
    return (cv2.resize(face_rgb, (128, 128)) / 255.0).reshape(1, 128, 128, 3)


def predict_face(face_rgb, emotion_model, age_gender_model):
    emotion_probs = emotion_model.predict(preprocess_for_emotion(face_rgb), verbose=0)[0]
    age_probs, gender_probs = age_gender_model.predict(
        preprocess_for_age_gender(face_rgb), verbose=0
    )
    emotion_idx = int(np.argmax(emotion_probs))
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
        "emotion_probs": emotion_probs,
    }


def analyze(image_rgb, models):
    emotion_model, age_gender_model, face_cascade = models
    image_rgb = image_rgb.copy()
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(100, 100))
    results = []
    for x, y, w, h in faces:
        face_crop = image_rgb[y:y+h, x:x+w]
        result = predict_face(face_crop, emotion_model, age_gender_model)
        result["box"] = (int(x), int(y), int(w), int(h))
        results.append(result)
    return results


def annotate(image_rgb, results):
    image = image_rgb.copy()
    for result in results:
        x, y, w, h = result["box"]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 220, 120), 3)
        label = f"{result['emotion_label']} | {result['age_label']} | {result['gender_label']}"
        cv2.rectangle(image, (x, max(0, y-30)), (x+min(w, 430), y), (15, 23, 42), -1)
        cv2.putText(image, label, (x+8, max(21, y-9)), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 2)
    return image


def result_card(i, result):
    st.markdown(f"### Face {i}")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Emotion", result["emotion_label"].title(), f"{result['emotion_conf']:.1f}%")
    with c2:
        st.metric("Age group", result["age_label"], f"{result['age_conf']:.1f}%")
    with c3:
        st.metric("Gender", result["gender_label"], f"{result['gender_conf']:.1f}%")
    labels = [x.title() for x in EMOTION_CLASSES]
    probs = result["emotion_probs"]
    st.caption("Emotion model output")
    st.bar_chart(dict(zip(labels, probs)), horizontal=True)

st.markdown('<div class="hero"><h1>🙂 Facial Emotion, Age Group & Gender Prediction</h1><p>Real-time computer vision using a custom CNN, MobileNetV2 transfer learning, OpenCV and Streamlit.</p></div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
for col, title, value in [(m1,"Emotion","59.4%"),(m2,"Age group","58.1%"),(m3,"Gender","87.2%"),(m4,"Emotion classes","7")]:
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-title">{title}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

st.info("These are probabilistic model outputs, not guaranteed truths. They should not be used to infer identity, personality, or actual demographic characteristics.", icon="ℹ️")
models = load_models()

tab_camera, tab_upload = st.tabs(["📷 Camera snapshot", "🖼️ Upload image"])
source_image = None
with tab_camera:
    camera_file = st.camera_input("Take a snapshot")
    if camera_file is not None:
        source_image = Image.open(camera_file).convert("RGB")
with tab_upload:
    uploaded_file = st.file_uploader("Upload JPG, JPEG or PNG", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        source_image = Image.open(uploaded_file).convert("RGB")

if source_image is not None:
    image_rgb = np.array(source_image)
    results = analyze(image_rgb, models)
    annotated = annotate(image_rgb, results)
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Detection")
        st.image(annotated, use_container_width=True)
        if results:
            st.success(f"Detected {len(results)} face(s).")
        else:
            st.warning("No face detected. Try a clear, front-facing image with good lighting.")
    with right:
        st.subheader("Analysis results")
        if results:
            for i, result in enumerate(results, 1):
                result_card(i, result)
        else:
            st.caption("Predictions will appear here after a face is detected.")
else:
    st.markdown("### Try the app")
    st.write("Use the camera or upload a photo to run the trained models on your own image.")

st.divider()
st.caption("Built as a Deep Learning course project at NED University of Engineering & Technology. Validation results shown above are from the project evaluation.")
