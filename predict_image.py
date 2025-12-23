import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load model
model = load_model("model/trash_model.keras")


# Class names must match the folders you used
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def predict_image(img_path):
    # Load and preprocess image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    # Output
    print(f"Predicted class: {class_names[class_index]} ({confidence*100:.2f}%)")

# Example usage
predict_image("/Users/amiraitayev/Documents/Trash_detection /Trash_detection /test_image/cardboard1.jpg")
