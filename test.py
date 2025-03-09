import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

test_path = "ck+fer_test"

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(48, 48),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=False  
)

model = load_model("VGG16.h5")

print("✅ Model test ediliyor...")
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\n✅ Test Doğruluğu: {test_accuracy:.4f}")
print(f"❌ Test Kayıp Değeri: {test_loss:.4f}")