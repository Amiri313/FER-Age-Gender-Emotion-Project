# Facial Emotion, Age Group, and Gender Prediction

Deep Learning course project — Post Graduate Diploma in Data Science & AI, NED University of Engineering & Technology.

**Jahangir Amiri, Sadaf Aslam** · Instructor: Sir Sajid Majeed

Real-time facial analysis from a webcam or photo:

- **Emotion** (7-class) — custom CNN, FER2013
- **Age group** (7-class) + **gender** — MobileNetV2 transfer learning on UTKFace, fine-tuned
- **Face detection** — OpenCV Haar Cascade

> Predictions are probabilistic model outputs. Not a reliable indicator of identity, personality, or actual demographic characteristics.

## Results

| Task | Validation accuracy |
|---|---|
| Emotion | 59.4% |
| Age group | 58.1% |
| Gender | 87.2% |

![FER2013 validation confusion matrix](assets/emotion_confusion_matrix.png)

Full breakdown: [PROJECT_REPORT.md](PROJECT_REPORT.md)

## Structure

```text
FER-Age-Gender-Emotion-Project/
├── README.md
├── PROJECT_REPORT.md
├── DEPLOYMENT.md
├── requirements.txt
├── .gitignore
├── haarcascade_frontalface_default.xml
├── assets/
│   └── (charts extracted from the notebook)
├── models/
│   ├── README.md
│   ├── emotion_model.keras
│   └── age_gender_model_finetuned.keras
├── notebooks/
│   └── Facial_Emotion_Age_Gender_Prediction.ipynb
├── src/
│   └── webcam_app.py
└── streamlit_app/
    ├── streamlit_app.py
    └── requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python src/webcam_app.py
python src/webcam_app.py --camera 1
python src/webcam_app.py --image photo.jpg
```

## Web demo

```bash
streamlit run streamlit_app/streamlit_app.py
```

Deployment: [DEPLOYMENT.md](DEPLOYMENT.md)

## Notebook

`notebooks/Facial_Emotion_Age_Gender_Prediction.ipynb` — dataset prep, training, evaluation, fine-tuning, export. Built for Colab, downloads datasets via Kaggle API.
