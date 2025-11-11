## tienes que ir a cd C:\Users\LG\Documents\DatasetEmociones y correlo en terminal mejor 

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# --- 1. Cargar el Modelo y las Etiquetas ---

# Cargar el modelo CNN que entrenamos
try:
    emotion_model = load_model('emotion_model_cnn.h5')
except Exception as e:
    print(f"Error cargando el modelo: {e}")
    print("Asegúrate de que 'emotion_model_cnn.h5' esté en la misma carpeta.")
    exit()

# Cargar el clasificador de caras (el que usaste al principio)
face_cascade_path = 'C:/Users/LG/Documents/inteligencia artificial/haarcascade_frontalface_alt.xml'
face_detector = cv2.CascadeClassifier(face_cascade_path)

# Cargar los nombres de las etiquetas
try:
    with open('emotion_labels.txt', 'r') as f:
        class_labels = [line.strip() for line in f]
except FileNotFoundError:
    print("Error: No se encontró 'emotion_labels.txt'.")
    exit()

print("Modelo, clasificador y etiquetas cargados correctamente.")

# --- 2. Iniciar la Cámara ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se puede abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1) # Espejo
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convertir a gris para el detector Haar
    
    # --- 3. Detectar Caras ---
    faces = face_detector.detectMultiScale(gray, 
                                           scaleFactor=1.1, 
                                           minNeighbors=5, 
                                           minSize=(30, 30))

    # Recorrer cada cara detectada
    for (x, y, w, h) in faces:
        # Dibujar el rectángulo de la cara
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # --- 4. Procesar la Cara para la CNN ---
        # Extraer la región de la cara (ROI) en escala de grises
        roi_gray = gray[y:y+h, x:x+w]
        
        # El modelo se entrenó con imágenes de 48x48
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        # Convertir a un formato que la CNN entienda
        roi = roi_gray.astype('float') / 255.0  # Normalizar (0 a 1)
        roi = img_to_array(roi)
        # La CNN espera un lote de imágenes: (1, 48, 48, 1)
        # 1 imagen, 48x48 de tamaño, 1 canal (gris)
        roi = np.expand_dims(roi, axis=0) 

        # --- 5. Predecir la Emoción ---
        prediction = emotion_model.predict(roi, verbose=0)[0]
        
        # Obtener el nombre de la emoción y la confianza
        emotion_label = class_labels[prediction.argmax()]
        confidence = prediction.argmax()
        
        # Poner el texto de la emoción
        texto = f"{emotion_label.capitalize()}" # capitalize() pone la primera letra en mayúscula
        cv2.putText(frame, texto, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --- 6. Mostrar el resultado ---
    cv2.imshow('Tu Detector de Emociones (CNN)', frame)
    
    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 7. Limpieza ---
cap.release()
cv2.destroyAllWindows()


