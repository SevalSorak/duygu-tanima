import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import VGG16, MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# CNN Modeli Oluşturma
def create_cnn_model(input_shape=(48, 48, 1), num_classes=4):
    """Sıfırdan bir CNN modeli oluşturur."""
    model = Sequential()

    # 1. Konvolüsyon Katmanı
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    # 2. Konvolüsyon Katmanı
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    # 3. Konvolüsyon Katmanı
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    # Flatten ve Fully Connected Katmanları
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Overfitting'i önlemek için
    model.add(Dense(num_classes, activation='softmax')) 

    # Modeli derle
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Transfer Learning Modeli (VGG16 veya MobileNetV2)
def create_transfer_learning_model(base_model_name="VGG16", input_shape=(48, 48, 3), num_classes=4):
    """
    Transfer learning modeli oluşturur. 
    base_model_name: "VGG16" veya "MobileNetV2"
    """
    if base_model_name == "VGG16":
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    elif base_model_name == "MobileNetV2":
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        raise ValueError("Geçersiz model adı! Lütfen 'VGG16' veya 'MobileNetV2' seçin.")

    # Önceden eğitilmiş modelin katmanlarını dondur (Freeze)
    for layer in base_model.layers:
        layer.trainable = False

    # Yeni katmanlar ekleyelim
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output_layer)

    # Modeli derle
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    return model