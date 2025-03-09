import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

LABELS = ["angry", "happy", "sad", "fear"]
LABEL_DICT = {label: i for i, label in enumerate(LABELS)}

def preprocess_image(img_path, target_size=(48, 48)):
    """CK+ ve FER veri setlerine farklı ön işleme uygular."""
    img = cv2.imread(img_path)

    if img is None:
        print(f"Hata: Görüntü yüklenemedi -> {img_path}")
        return None

    dataset = "FER"
    if img_path.split("/")[-1].startswith("S0"):  
        dataset = "CK"

    if dataset == "FER":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Gri tonlamaya çevir
        img = cv2.GaussianBlur(img, (3, 3), 0)  # Gürültüyü azalt

        # Kontrast artırma (Adaptive Histogram Equalization - AHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # CK+48 için renkli tut

    img = cv2.resize(img, target_size)

    img = img / 255.0

    if dataset == "FER":
        img = img.reshape(target_size[0], target_size[1], 1)  # FER gri tonlamalı olduğu için 1 kanal
    else:
        img = img.reshape(target_size[0], target_size[1], 3)  # CK+ renkli olduğu için 3 kanal

    return img

def load_dataset(dataset_path, labels=LABELS):
    """Eğitim ve test verisini yükler, ön işler ve NumPy dizisi olarak döndürür."""
    images = []
    image_labels = []

    for label in labels:
        class_path = os.path.join(dataset_path, label)
        if not os.path.exists(class_path):
            print(f"Hata: {class_path} bulunamadı, etiketi atlıyorum...")
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = preprocess_image(img_path)  # 🔹 `color_mode` kaldırıldı

            if img is not None:
                images.append(img)
                image_labels.append(LABEL_DICT[label])

    images = np.array(images)
    image_labels = to_categorical(np.array(image_labels), num_classes=len(labels))

    print(f"{dataset_path} veri seti: {images.shape[0]} örnek yüklendi.")

    return images, image_labels

if __name__ == "__main__":
    train_path = "ck+fer_train"  
    test_path = "ck+fer_test"    

    X_train, y_train = load_dataset(train_path)
    X_test, y_test = load_dataset(test_path)

    print(f"✅ Veri başarıyla yüklendi! Eğitim seti: {X_train.shape}, Test seti: {X_test.shape}")
