import cv2 as cv
import numpy as np
import os



personas = ['Pedro', 'Obed', 'Eliseo', 'Sebas']
data_path = r'C:\Users\LG\Downloads\sebas' # 'r' al inicio para manejar bien las \

# Listas para guardar las caras y sus etiquetas
training_data = []
training_labels = []

# Clasificador de caras (el mismo que ya tenías)
rostro_cascade = cv.CascadeClassifier('C:/Users/LG/Documents/inteligencia artificial/haarcascade_frontalface_alt.xml')

# Instanciar el reconocedor
face_recognizer = cv.face.LBPHFaceRecognizer_create()

print("Preparando datos de entrenamiento...")

# Recorrer cada carpeta de persona
for label, persona in enumerate(personas):
    person_path = os.path.join(data_path, persona)
    print(f'Procesando imágenes de: {persona}')

    # Recorrer cada imagen en la carpeta de la persona
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        
        # Leer la imagen en escala de grises
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f'No se pudo leer la imagen: {img_name}')
            continue
        
        # Detectar la cara en la imagen de entrenamiento
        rostros = rostro_cascade.detectMultiScale(img, 1.1, 4)
        
        # Asumimos que cada imagen de entrenamiento tiene UNA cara
        for (x, y, w, h) in rostros:
            face_roi = img[y:y+h, x:x+w] # Recortar la cara
            training_data.append(face_roi)
            training_labels.append(label) # 'label' es el número (0=pedro, 1=obed, etc.)

print("Datos preparados.")
print("Entrenando el modelo...")

training_labels = np.array(training_labels)
face_recognizer.train(training_data, training_labels)
face_recognizer.save('face_model.yml')

print("¡Modelo entrenado y guardado como 'face_model.yml'!")