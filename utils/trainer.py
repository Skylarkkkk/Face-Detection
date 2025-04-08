import os
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import time


def train_model():
    faces = []
    labels = []
    label_ids = {}
    current_id = 0
    data_dir = "./data"
    model_dir = "./models"

    for file in os.listdir(data_dir):
        if file.endswith(('.jpg', '.png')):
            name = ''.join(filter(str.isalpha, file.split('.')[0]))
            if name not in label_ids:
                label_ids[name] = current_id
                current_id += 1
            label = label_ids[name]
            path = os.path.join(data_dir, file)
            image = Image.open(path).convert('L')
            image_np = np.array(image, 'uint8')
            detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces_detected = detector.detectMultiScale(image_np)
            for (x, y, w, h) in faces_detected:
                print(f"Detected face for {name} in {file}: x={x}, y={y}, w={w}, h={h}")
                faces.append(image_np[y:y+h, x:x+w])
                labels.append(label)

    if not faces:
        raise Exception("No faces found. Check your data.")

    model = cv2.face.LBPHFaceRecognizer_create()
    print("Training model...")
    model.train(faces, np.array(labels))
    os.makedirs(model_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model.save(os.path.join(model_dir, f"face_model_{timestamp}.yml"))
    # Save the last model
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "face_model_last.yml"))

    # Save name mappings
    df = pd.DataFrame(list(label_ids.items()), columns=['name', 'label'])
    df.to_csv(os.path.join(data_dir, 'names.csv'), index=False)

    print("Training complete.")
    print(f"Model saved to {model_dir}")