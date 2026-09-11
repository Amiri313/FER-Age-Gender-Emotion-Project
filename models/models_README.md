# Trained models

- `emotion_model.keras` — custom CNN, FER2013
- `age_gender_model_finetuned.keras` — MobileNetV2, fine-tuned on UTKFace; used by `webcam_app.py` and `streamlit_app.py`

Both files are committed directly (under 30 MB combined). Past ~50–100 MB, switch to Git LFS or external hosting (see `DEPLOYMENT.md`).
