import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model and class names
model = load_model("model/trash_model.keras")
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def predict_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Skipped: {img_path}")
        return

    img_resized = cv2.resize(img, (224, 224))
    img_resized = img_resized.astype('float32') / 255.0
    img_expanded = np.expand_dims(img_resized, axis=0)

    prediction = model.predict(img_expanded)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    print(f"✅ {os.path.basename(img_path)} → {class_names[class_index]} ({confidence*100:.2f}%)")

# Folder of test images
test_folder = "dataset"

for filename in os.listdir(test_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(test_folder, filename)
        predict_image(img_path)
