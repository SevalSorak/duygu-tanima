import os
import numpy as np
import shutil
import random
from collections import defaultdict

dataset_path = r"C:\Users\PC\Downloads\archive\CK+48"

labels = ["angry", "fear", "happy", "sad"]

person_images = defaultdict(list)

for label in labels:
    class_path = os.path.join(dataset_path, label)
    
    for img in os.listdir(class_path):
        img_path = os.path.join(class_path, img)

        if "S" in img and "_" in img:  
            person_id = img.split("_")[0]  
        else:  
            person_id = "FER_" + img.split(".")[0]  

        person_images[person_id].append((img_path, label))

person_ids = list(person_images.keys())
random.seed(42)
random.shuffle(person_ids)

split_idx = int(len(person_ids) * 0.8)  
train_ids = person_ids[:split_idx]
test_ids = person_ids[split_idx:]

train_images, test_images = [], []

for person_id in train_ids:
    train_images.extend(person_images[person_id])

for person_id in test_ids:
    test_images.extend(person_images[person_id])

print(f"Eğitim seti: {len(train_images)} görüntü")
print(f"Test seti: {len(test_images)} görüntü")

def save_split_data(split_images, split_name):
    """Train ve test verilerini yeni dizinlere kopyalar."""
    split_path = f"ck+fer_{split_name}"
    os.makedirs(split_path, exist_ok=True)

    for label in labels:
        os.makedirs(os.path.join(split_path, label), exist_ok=True)

    for img_path, label in split_images:
        shutil.copy(img_path, os.path.join(split_path, label))

save_split_data(train_images, "train")
save_split_data(test_images, "test")

print("✅ CK+48 ve FER2013 verisi başarıyla eğitim ve test setine ayrıldı!")
