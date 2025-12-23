import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("model/trash_model.keras")

# Class labels (must match your dataset)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Open webcam (0 = default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Prepare frame for prediction
    img = cv2.resize(frame, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img)
    class_index = np.argmax(preds)
    confidence = np.max(preds)

    # Overlay label on frame
    label = f"{class_names[class_index]} ({confidence * 100:.2f}%)"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2, cv2.LINE_AA)

    # Display the result
    cv2.imshow("Trash Detector", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
