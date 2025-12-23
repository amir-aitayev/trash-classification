import matplotlib.pyplot as plt
import cv2
import random
import os

dataset_path = "dataset"

for class_name in sorted(os.listdir(dataset_path)):
    class_path = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_path):
        num_images = len([
            f for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        print(f"{class_name:<10} → {num_images} images")
samples_per_class = 3

# Preview Sample Images per Class
for class_name in sorted(os.listdir(dataset_path)):
    class_path = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_path):
        image_files = [f for f in os.listdir(class_path) if f.endswith((".jpg", ".jpeg", ".png"))]
        chosen = random.sample(image_files, min(samples_per_class, len(image_files)))

        for img_file in chosen:
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.imshow(img)
                plt.title(f"{class_name}: {img_file}")
                plt.axis('off')
                plt.show()

