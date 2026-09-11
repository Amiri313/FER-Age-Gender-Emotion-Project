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
6. Optionally choose a custom app subdomain such as `fer-age-gender-emotion` if it is available.
7. Click **Deploy**.

Streamlit Community Cloud creates a public `streamlit.app` URL for the deployment. After deployment, add that URL to the **Live Streamlit demo** section of `README.md`.

> The repository owner must authorize Streamlit to access the GitHub repository. This is an account-level authorization step and cannot be completed by the GitHub repository integration alone.

### After deployment

Any future push to `main` will be picked up by the deployed app automatically. If dependencies change, the deployment may take a few minutes to rebuild.

## Model files

`models/emotion_model.keras` and `models/age_gender_model_finetuned.keras` are committed directly and are currently small enough for the repository. For substantially larger future models, use Git LFS or external model storage.

## Notes

- The Streamlit app uses `opencv-python-headless`, while the local webcam app uses the regular `opencv-python` package.
- `.streamlit/config.toml` provides the app theme and headless server configuration.
