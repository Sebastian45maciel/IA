import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import (
    BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
)
from keras.layers import LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator # ¡Esto es clave!

# --- 1. Definir Rutas ---
# ¡Actualizado! Ahora usamos las carpetas 'train' y 'test' que descargaste.
train_dir = 'train'
validation_dir = 'test' 

# Las imágenes de este dataset son de 48x48 en escala de grises.
IMG_HEIGHT = 48
IMG_WIDTH = 48

# --- 2. Parámetros de Entrenamiento ---
INIT_LR = 1e-3 # Tasa de aprendizaje
epochs = 30    # 30 épocas es un buen comienzo
batch_size = 64 # Imágenes por lote

# --- 3. Crear Generadores de Imágenes ---
# Esto lee las imágenes directamente de tus carpetas 'train' y 'test'
# y les aplica "aumentación de datos" (las gira, voltea, etc.)

print("Configurando generador de entrenamiento (train)...")
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalizar (dividir entre 255)
    rotation_range=20,       # Gira la imagen
    width_shift_range=0.1,   # Mueve la imagen horizontalmente
    height_shift_range=0.1,  # Mueve la imagen verticalmente
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,    # Voltea la imagen
    fill_mode='nearest'
)

print("Configurando generador de validación (test)...")
validation_datagen = ImageDataGenerator(rescale=1./255) # Al de validación solo lo normalizamos

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    color_mode='grayscale', # ¡Clave! Las imágenes son de 48x48 en grises
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    color_mode='grayscale', # ¡Clave!
    class_mode='categorical',
    shuffle=False
)

# Obtenemos el número de clases (deberían ser 7) automáticamente
nClasses = len(train_generator.class_indices)
print(f"Clases encontradas: {nClasses}")
print(f"Mapeo de clases: {train_generator.class_indices}")


# --- 4. Crear el Modelo de CNN (La arquitectura de tu profe) ---
# (Adaptado para imágenes de 48x48 en 1 canal de color)

emotion_model = Sequential()

# Capa 1
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same', 
                         input_shape=(IMG_HEIGHT, IMG_WIDTH, 1))) # 1 canal (grises)
emotion_model.add(LeakyReLU(alpha=0.1))
emotion_model.add(MaxPooling2D((2, 2), padding='same'))
emotion_model.add(Dropout(0.25))

# Capa 2
emotion_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
emotion_model.add(LeakyReLU(alpha=0.1))
emotion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
emotion_model.add(Dropout(0.25))

# Capa 3
emotion_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
emotion_model.add(LeakyReLU(alpha=0.1))
emotion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
emotion_model.add(Dropout(0.25))

# Capa 4 - Aplanar y Conectar
emotion_model.add(Flatten())
emotion_model.add(Dense(128, activation='linear')) # Capa densa más grande
emotion_model.add(LeakyReLU(alpha=0.1))
emotion_model.add(Dropout(0.3))

# Capa de Salida
emotion_model.add(Dense(nClasses, activation='softmax')) # nClasses será 7

emotion_model.summary()

# --- 5. Compilar el Modelo ---
emotion_model.compile(loss='categorical_crossentropy', 
                      optimizer=tf.keras.optimizers.Adam(learning_rate=INIT_LR),
                      metrics=['accuracy'])

# --- 6. Entrenar el Modelo ---
print("\n--- Iniciando Entrenamiento ---")
print("Esto puede tardar varios minutos (o más)...")

history = emotion_model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // batch_size,
    verbose=1
)

# --- 7. Guardar el Modelo Entrenado ---
model_save_path = 'emotion_model_cnn.h5' # Le pongo 'cnn' para diferenciarlo
emotion_model.save(model_save_path)

# --- 8. Guardar el mapeo de clases ---
# Guardamos los nombres de las clases en un archivo de texto para usarlos después
class_names = list(train_generator.class_indices.keys())
with open('emotion_labels.txt', 'w') as f:
    for name in class_names:
        f.write(f"{name}\n")

print(f"\n--- ¡Entrenamiento Finalizado! ---")
print(f"Modelo guardado en: {model_save_path}")
print(f"Etiquetas guardadas en: emotion_labels.txt")