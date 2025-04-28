# Multi-Modal-Medical-Image-Diagnostic-System

This project focuses on classifying MRI brain images to detect tumors using transfer learning and classical ML approaches.

## Dataset
- Source: [Brain MRI Images for Brain Tumor Detection (Kaggle)](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

## Technologies Used
- Python
- PyTorch
- TensorFlow (optional reference)
- scikit-learn
- NumPy
- Pandas
- MLflow (for future experiment tracking)

## Approach
1. Data preprocessing (Resize, Grayscale to RGB, Normalize)
2. Model experimentation:
   - ResNet18 (transfer learning)
   - SVM and Logistic Regression (feature-based)
3. Evaluation and comparison

## Results
- ResNet18 achieved ~92%+ validation accuracy.
- SVM and Logistic Regression provided baseline comparisons.


