# Deployment guide

## Recommended setup

Use **GitHub** as the source-code/project portfolio and **Hugging Face Spaces** as the public interactive demo. This separation is practical because the GitHub repository can document the complete academic work while the Space can focus on a simple user-facing demo.

### Option A — Local webcam demo

Best for your NED demonstration and portfolio video.

```bash
pip install -r requirements.txt
python src/webcam_app.py
```

This version accesses the computer's webcam directly. Run it from the project root — the script resolves the model files in `models/` and the Haar Cascade file automatically.

If you don't have a webcam available (or just want a quick, clean screenshot for the README), run it against a single photo instead:

```bash
python src/webcam_app.py --image path/to/photo.jpg
```

This saves an annotated copy next to the original — a fast way to produce the demo screenshots the README still needs.

### Option B — Public web demo

For a public browser demo, create a Hugging Face Space using **Gradio** or another browser-friendly interface. A web deployment cannot simply use the desktop OpenCV webcam loop in `webcam_app.py`; the browser needs to provide the camera frames to the application.

A practical public demo should accept a webcam/image frame and return the predicted emotion, age group, and gender. Keep the demo lightweight and avoid collecting or storing user images.

## Recommended deployment order

1. Finish and execute the notebook in Colab.
2. Download `emotion_model.keras` and `age_gender_model_finetuned.keras`.
3. Test the local webcam application (`python src/webcam_app.py`).
4. Use `--image` on a couple of test photos, or record a short 20–40 second webcam demo, showing the predictions.
5. Create the GitHub repository and upload the notebook, Python app, requirements, README, report, models folder, and Haar Cascade file.
6. Both trained model files are small enough (under 30 MB combined) to commit directly, which is what this repo already does — no Git LFS or external hosting is needed at the current size. Revisit this only if you retrain with a larger backbone and the files grow past roughly 50–100 MB.
7. Create a Hugging Face Space for the browser demo.
8. Put the GitHub and live-demo links on LinkedIn.

## GitHub model-file advice

Normal GitHub repositories have file-size limits (100 MB hard limit per file). Do not commit raw datasets, Colab caches, or every training checkpoint. If a trained model grows too large for a plain commit, use Git LFS or a model-hosting service instead.

## Academic submission vs public portfolio

For the NED submission, keep the complete notebook and report available to the instructor. For GitHub, keep the repository reproducible but avoid committing private credentials, Kaggle keys, raw datasets, or unnecessary intermediate checkpoints.
