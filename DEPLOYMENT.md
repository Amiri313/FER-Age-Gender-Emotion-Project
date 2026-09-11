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

## Web — Streamlit Community Cloud (free)

```bash
streamlit run streamlit_app/streamlit_app.py
```

Deploy: push `streamlit_app/` → share.streamlit.io → connect repo → main file path `streamlit_app/streamlit_app.py`.

`streamlit_app/requirements.txt` uses `opencv-python-headless`, separate from the root `requirements.txt` (`opencv-python`, for local `cv2.imshow`).

## Web — Hugging Face Spaces (paid)

Gradio version: `app.py`, separate Space repo. Requires HF PRO ($9/mo) to create a Gradio or Docker Space.

## Model files

`models/emotion_model.keras`, `models/age_gender_model_finetuned.keras` — committed directly (<30 MB combined). Git LFS or external hosting past ~50–100 MB.
