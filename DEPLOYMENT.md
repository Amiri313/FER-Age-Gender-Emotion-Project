# Deployment

## Local

```bash
pip install -r requirements.txt
python src/webcam_app.py
```

Single image instead of webcam:

```bash
python src/webcam_app.py --image path/to/photo.jpg
```

## Streamlit Community Cloud

This repository is structured for Streamlit Community Cloud. The Streamlit entrypoint is:

```text
streamlit_app/streamlit_app.py
```

The app-specific dependencies are kept beside the entrypoint in:

```text
streamlit_app/requirements.txt
```

### Deploy

1. Sign in to **Streamlit Community Cloud** with the GitHub account that owns this repository.
2. Choose **Create app**.
3. Select this repository: `Amiri313/FER-Age-Gender-Emotion-Project`.
4. Branch: `main`.
5. Main file path: `streamlit_app/streamlit_app.py`.
6. Select **Python 3.11** in Advanced settings for compatibility with the pinned TensorFlow version used by the Streamlit app.
7. Optionally choose a custom app subdomain such as `fer-age-gender-emotion` if it is available.
8. Click **Deploy**.

The current public deployment is:

https://fer-age-gender-emotion.streamlit.app/

### Application modes

- **Local OpenCV app:** continuous real-time webcam inference using `cv2.VideoCapture()`.
- **Streamlit app:** browser-based camera snapshots and image upload. `st.camera_input()` captures a still image rather than providing a continuous video stream.

### After deployment

Any future push to `main` will be picked up by the deployed app automatically. If dependencies change, the deployment may take a few minutes to rebuild.

## Model files

`models/emotion_model.keras` and `models/age_gender_model_finetuned.keras` are committed directly and are currently small enough for the repository. For substantially larger future models, use Git LFS or external model storage.

## Notes

- The Streamlit app uses `opencv-python-headless`, while the local webcam app uses the regular `opencv-python` package.
- `.streamlit/config.toml` provides the app theme and headless server configuration.
