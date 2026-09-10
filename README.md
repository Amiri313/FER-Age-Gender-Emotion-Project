# Facial Emotion, Age Group, and Gender Prediction

## Deep Learning Course Project

### Project Partners
- **Jahangir Amiri**
- **Sadaf Aslam**

**Course:** Deep Learning
**Instructor:** Sir Sajid Majeed
**Program:** Post Graduate Diploma (PGD) in Data Science & AI
**Institution:** NED University of Engineering & Technology

---

A computer-vision project that combines two deep-learning models to perform real-time facial analysis from a webcam:

- **Facial emotion classification** using a custom CNN trained on FER2013.
- **Age-group and gender classification** using MobileNetV2 transfer learning on UTKFace, followed by fine-tuning.
- **Face detection** using OpenCV's Haar Cascade classifier.

The notebook documents dataset inspection, exploratory analysis, preprocessing, model development, training, evaluation, fine-tuning, and model saving.

> **Important:** Age, gender, and emotion predictions from facial images are probabilistic model outputs and can be inaccurate. They should not be treated as reliable judgments about a person's identity, personality, or actual demographic characteristics.

## Results

| Task | Validation accuracy |
|---|---|
| Emotion (7-class, FER2013) | 59.4% |
| Age group (7-class, UTKFace, fine-tuned) | 58.1% |
| Gender (binary, UTKFace, fine-tuned) | 87.2% |

Full breakdown, confusion matrix discussion, and per-emotion-class analysis are in [PROJECT_REPORT.md](PROJECT_REPORT.md).

## Project structure

```text
FER-Age-Gender-Emotion-Project/
├── README.md
├── PROJECT_REPORT.md
├── DEPLOYMENT.md
├── requirements.txt
├── .gitignore
├── haarcascade_frontalface_default.xml
├── models/
│   ├── README.md
│   ├── emotion_model.keras
│   └── age_gender_model_finetuned.keras
├── notebooks/
│   └── Facial_Emotion_Age_Gender_Prediction.ipynb
└── src/
    └── webcam_app.py
```

## Models

Both trained models are included in this repository under `models/`:

```text
models/emotion_model.keras
models/age_gender_model_finetuned.keras
```

`age_gender_model_finetuned.keras` is the model actually used by the webcam app — it's the MobileNetV2 model after the fine-tuning stage described in the report. The intermediate frozen-backbone version is not kept in the repo, only its metrics (see PROJECT_REPORT.md, section 5.2).

## Run locally

### 1. Create an environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the webcam application

Run this from the project root — `webcam_app.py` resolves the model and cascade paths relative to the repo root automatically, so nothing needs to be copied around:

```bash
python src/webcam_app.py
```

Press **q** to close the window. If you have more than one camera, pick a different index with `--camera 1`. If you don't have a webcam handy (e.g. for grading), you can run it against a single photo instead:

```bash
python src/webcam_app.py --image path/to/photo.jpg
```

This saves an annotated copy (`photo_predicted.jpg`) next to the original.

## Notebook

Open `notebooks/Facial_Emotion_Age_Gender_Prediction.ipynb` in Jupyter Notebook or upload it to Google Colab. The notebook downloads the datasets through Kaggle, performs exploratory checks and sample visualization, trains the models, evaluates them, and saves the trained models.

## Suggested GitHub presentation

Keep the repository focused on the project rather than uploading raw datasets or every training checkpoint. A good public repository contains the notebook, application code, setup instructions, model information, and a short report.

For the strongest presentation, add 2–4 screenshots or a short demo GIF/video under `assets/` after testing the application locally — the `--image` mode above is a quick way to generate a clean annotated screenshot for this.
