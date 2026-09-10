# Project Report

## Facial Emotion, Age Group, and Gender Prediction Using Deep Learning

### Project Partners

**Jahangir Amiri**  
**Sadaf Aslam**

**Course:** Deep Learning  
**Instructor:** Sir Sajid Majeed  
**Program:** Post Graduate Diploma (PGD) in Data Science & AI  
**Institution:** NED University of Engineering & Technology

---

## 1. Project Objective

The objective of this project is to develop a computer-vision system that detects a face from an image or webcam frame and produces three model-based predictions: facial emotion, age group, and gender. The project demonstrates the use of convolutional neural networks, transfer learning, multi-output modelling, and fine-tuning for facial-image analysis.

## 2. Datasets

### FER2013

FER2013 is used for seven-class facial emotion classification. The notebook first inspects the dataset and displays sample images before creating the training and validation generators.

### UTKFace

UTKFace is used for age and gender prediction. Age and gender labels are extracted from the image filenames, and the original ages are converted into seven broader age brackets for classification.

## 3. Methodology

### Emotion Model

A custom convolutional neural network is trained on 48×48 grayscale facial images. Batch normalization, pooling, dropout, data augmentation, class weighting, early stopping, model checkpointing, and learning-rate reduction are used to improve training stability and generalization.

### Age and Gender Model

MobileNetV2 pretrained on ImageNet is used as the feature extractor. The model has two outputs: a seven-class age-group classifier and a binary gender classifier. The pretrained base is initially frozen and is then partially unfrozen for fine-tuning with a smaller learning rate.

### Face Detection

OpenCV's Haar Cascade frontal-face detector is used in the local webcam application before the detected face is cropped and passed to the trained models.

## 4. Exploratory Data Analysis

The notebook includes basic dataset inspection before model training. For FER2013, the class structure and representative images are checked. For UTKFace, sample images are displayed with labels derived from the filenames, and the distribution of the seven age groups is examined.

## 5. Evaluation

The final evaluation values should be taken directly from the executed notebook so that the report matches the actual training run. The main results to record are:

| Task | Metric | Final result |
|---|---|---|
| Emotion classification | Validation accuracy | Add value from notebook |
| Emotion classification | Validation loss | Add value from notebook |
| Age-group classification | Validation accuracy | Add value from notebook |
| Gender classification | Validation accuracy | Add value from notebook |
| Fine-tuned age-group model | Validation accuracy | Add value from notebook |
| Fine-tuned gender model | Validation accuracy | Add value from notebook |

The notebook's saved models are intended for later local testing through the webcam application.

## 6. Limitations

The predictions are affected by lighting, pose, occlusion, image quality, facial-expression ambiguity, dataset bias, and the relatively coarse age-group labels. Gender prediction is also a model classification task and should not be presented as a definitive determination of a person's gender identity.

## 7. Future Improvements

Possible improvements include stronger data balancing, a more capable face detector, better age-label modelling, confusion-matrix analysis, calibration of prediction confidence, additional augmentation, and comparison with modern pretrained vision backbones.

## 8. Conclusion

This project demonstrates a complete deep-learning workflow from dataset preparation and exploratory analysis through model training, evaluation, fine-tuning, and deployment-oriented testing. The combination of a custom CNN for emotion recognition and MobileNetV2 transfer learning for age-group and gender prediction provides a practical example of applying deep learning to real-time facial-image analysis.
