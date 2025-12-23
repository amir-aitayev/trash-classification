# Trash Classification Using CNN and MobileNetV2

## 👤 Author
Amir Aitayev  


## 🎯 Project Overview
This project implements a deep learning–based image classification system for automatic waste sorting into **six categories**:
**cardboard, glass, metal, paper, plastic, and trash**.

The project compares a **custom Convolutional Neural Network (CNN)** trained from scratch with a **transfer learning approach using MobileNetV2**, pretrained on ImageNet. The results demonstrate that transfer learning significantly improves classification accuracy, especially for visually similar and underrepresented waste classes.

## 🧠 Key Features
- Image classification using deep learning
- Transfer learning with **MobileNetV2 (ImageNet)**
- Custom baseline CNN for comparison
- Real-time data augmentation to address class imbalance
- Model evaluation using confusion matrices and classification reports
- Multiple inference modes: single image, batch processing, and live webcam input

## 📁 Project Structure
train_model.py — Train the MobileNetV2 transfer learning model
baseline_model.py — Train and evaluate baseline CNN model
predict_image.py — Predict the class of a single image
batch_predict.py — Perform batch prediction on image folders
webcam_predict.py — Real-time trash classification using webcam
evaluate_model.py — Generate confusion matrix and evaluation metrics
inspect_dataset.py — Inspect dataset structure and sample images
data_loader.py — Dataset loading and preprocessing utilities
check_setup.py — Environment and dependency validation
main.py — Entry point for running the project


## 📊 Dataset
The dataset consists of waste images grouped into six categories:
cardboard, glass, metal, paper, plastic, and trash.

To improve generalization and reduce bias caused by class imbalance, **real-time data augmentation** was applied, including rotation, zooming, shifting, and horizontal flipping.

Dataset source:  
https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification

## ⚙️ Setup Instructions
```bash
git clone https://github.com/amir-aitayev/trash-classification.git
cd trash-classification
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Place the dataset inside a dataset/ directory before training.
▶️ How to Run
python train_transfer.py        # Train the MobileNetV2 model
python predict_image.py         # Predict a single image
python batch_predict.py         # Batch prediction on folders
python webcam_predict.py        # Real-time webcam inference
python evaluate_model.py        # Generate evaluation metrics
📈 Results
Baseline CNN: ~78–80% validation accuracy
MobileNetV2: 90%+ validation accuracy
Transfer learning substantially improved classification performance, particularly for difficult and visually similar waste categories such as plastic, metal, and trash.
🛠️ Technologies Used
Python
TensorFlow / Keras
OpenCV
NumPy
scikit-learn
Matplotlib
Seaborn
🔮 Future Improvements
Expansion of the dataset for improved robustness
Optimization for deployment on embedded and edge devices
Integration of object detection models (e.g., YOLO, Faster R-CNN) for multi-object waste sorting
📌 License
This project is intended for educational and research purposes.
