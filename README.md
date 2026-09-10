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

## Project structure

```text
FER-Age-Gender-Emotion-Project/
├── emotion_age_prediction.ipynb
├── webcam_app.py
├── haarcascade_frontalface_default.xml
├── requirements.txt
├── requirements-demo.txt
├── requirements-local.txt
├── README.md
├── DEPLOYMENT.md
├── PROJECT_REPORT.md
├── .gitignore
└── models/
    └── README.md
```

## Models

For local testing, use:

```text
emotion_model.keras
age_gender_model_finetuned.keras
```

`age_gender_model.keras` is the earlier transfer-learning model before fine-tuning. It is optional for the final demo.

Model files are not included in this repository template because trained Keras files can be large. Add them locally if they are small enough for your chosen hosting method, or host them separately.

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

### 3. Put the trained models in the project folder

The simplest layout for the current `webcam_app.py` is:

```text
project/
├── webcam_app.py
├── emotion_model.keras
├── age_gender_model_finetuned.keras
└── haarcascade_frontalface_default.xml
```

### 4. Start the webcam application

```bash
python webcam_app.py
```

Press **q** to close the application.

## Notebook

Open `emotion_age_prediction.ipynb` in Jupyter Notebook or upload it to Google Colab. The notebook downloads the datasets through Kaggle, performs exploratory checks and sample visualization, trains the models, evaluates them, and saves the trained models.

## Suggested GitHub presentation

Keep the repository focused on the project rather than uploading raw datasets or every training checkpoint. A good public repository contains the notebook, application code, setup instructions, model information, and a short report.

For the strongest presentation, add 2–4 screenshots or a short demo GIF/video under `assets/` after testing the application locally.
