"""Run the trained models locally with a webcam, or on a single test image.

Expected repo layout (this file lives in src/):
    project_root/
    ├── haarcascade_frontalface_default.xml
    ├── models/
    │   ├── emotion_model.keras
    │   └── age_gender_model_finetuned.keras
    └── src/
        └── webcam_app.py   <- this file

Run (webcam):
    python src/webcam_app.py
    python src/webcam_app.py --camera 1        # use a different camera index

Run (single image, useful if you don't have a webcam handy for grading/demo):
    python src/webcam_app.py --image path/to/photo.jpg

Press q to quit the webcam window.
"""
import argparse
import os
import time

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- Path setup -------------------------------------------------------------
# This file lives in src/, but the trained models live in models/ and the
# Haar cascade lives at the project root — so paths must be resolved relative
# to the project root, not relative to this script's own folder.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

EMOTION_MODEL_PATH = os.path.join(MODELS_DIR, "emotion_model.keras")
AGE_GENDER_MODEL_PATH = os.path.join(MODELS_DIR, "age_gender_model_finetuned.keras")
CASCADE_PATH = os.path.join(PROJECT_ROOT, "haarcascade_frontalface_default.xml")

for path in (EMOTION_MODEL_PATH, AGE_GENDER_MODEL_PATH, CASCADE_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Check that the models/ folder and haarcascade file sit at the "
            "project root, one level above src/."
        )

emotion_model = load_model(EMOTION_MODEL_PATH)
age_gender_model = load_model(AGE_GENDER_MODEL_PATH)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
AGE_BRACKETS = ["0-12", "13-19", "20-29", "30-39", "40-49", "50-59", "60+"]
GENDER_LABELS = ["Male", "Female"]

# Below this confidence, draw the box in yellow instead of green so low-trust
# predictions are visually distinguishable rather than presented as certain.
CONFIDENCE_WARN_THRESHOLD = 50.0


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


def predict_face(face_crop):
    """Run both models on one cropped face and return a results dict."""
    emotion_probs = emotion_model.predict(preprocess_for_emotion(face_crop), verbose=0)[0]
    emotion_idx = int(np.argmax(emotion_probs))

    age_probs, gender_probs = age_gender_model.predict(
        preprocess_for_age_gender(face_crop), verbose=0
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


def draw_predictions(frame, x, y, w, h, result):
    min_conf = min(result["emotion_conf"], result["age_conf"], result["gender_conf"])
    color = (0, 255, 0) if min_conf >= CONFIDENCE_WARN_THRESHOLD else (0, 220, 255)

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        frame, f"{result['emotion_label']} ({result['emotion_conf']:.0f}%)",
        (x, max(25, y - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
    )
    cv2.putText(
        frame,
        f"{result['age_label']} | {result['gender_label']} "
        f"({result['age_conf']:.0f}% / {result['gender_conf']:.0f}%)",
        (x, max(45, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
    )


def detect_faces(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=8, minSize=(100, 100)
    )


def run_on_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not read image: {image_path}")
        return

    for x, y, w, h in detect_faces(frame):
        result = predict_face(frame[y:y + h, x:x + w])
        draw_predictions(frame, x, y, w, h, result)
        print(result)

    out_path = os.path.splitext(image_path)[0] + "_predicted.jpg"
    cv2.imwrite(out_path, frame)
    print(f"Saved annotated image to {out_path}")


def run_on_webcam(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera index {camera_index}.")
        return

    window_name = "Facial Emotion, Age & Gender Detection"
    previous_time = time.time()
    smoothed_fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        for x, y, w, h in detect_faces(frame):
            result = predict_face(frame[y:y + h, x:x + w])
            draw_predictions(frame, x, y, w, h, result)

        current_time = time.time()
        instant_fps = 1 / max(current_time - previous_time, 1e-6)
        previous_time = current_time
        # Exponential moving average so the on-screen FPS number doesn't jitter.
        smoothed_fps = instant_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * instant_fps
        cv2.putText(frame, f"FPS: {smoothed_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--image", type=str, default=None,
                         help="Path to a single image instead of using the webcam")
    args = parser.parse_args()

    if args.image:
        run_on_image(args.image)
    else:
        run_on_webcam(args.camera)


if __name__ == "__main__":
    main()
