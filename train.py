import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization

train_path = "ck+fer_train"
test_path = "ck+fer_test"

learning_rate = 0.00005  
batch_size = 32  
dropout_rate = 0.4  
dense_units = 128  
epochs = 50  

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=40,  
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.5, 1.5],
    shear_range=0.2,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(48, 48),
    color_mode="rgb",  
    batch_size=batch_size,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(48, 48),
    color_mode="rgb",  
    batch_size=batch_size,
    class_mode="categorical"
)

base_model = VGG16(weights="imagenet", include_top=False, input_shape=(48, 48, 3))

for layer in base_model.layers[:-12]:  # Son 12 katmanı aç
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(dense_units, activation="relu")(x)
x = BatchNormalization()(x)  # BatchNormalization şimdi Dense'den sonra
x = Dropout(dropout_rate)(x)  # Dropout doğru konumda
output_layer = Dense(4, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output_layer)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer,
              loss="categorical_crossentropy",
              metrics=["accuracy"])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

history = model.fit(train_generator,
                    epochs=epochs,
                    validation_data=test_generator,
                    callbacks=[early_stopping, reduce_lr])

model.save("VGG16.h5")

print("✅ Optimize edilmiş Fine-tuned VGG16 modeli kaydedildi!")