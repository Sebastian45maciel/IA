import numpy as np
import cv2 as cv

# El clasificador para detectar *dónde* hay una cara
rostro = cv.CascadeClassifier('C:/Users/LG/Documents/inteligencia artificial/haarcascade_frontalface_alt.xml')

# --- El Reconocedor que entrenamos ---
face_recognizer = cv.face.LBPHFaceRecognizer_create()
# Cargar el modelo entrenado
face_recognizer.read('face_model.yml') 

# --- Lista de nombres ---
# DEBE estar en el MISMO ORDEN que en el script de entrenamiento
names = ['Pedro', 'Obed', 'Eliseo', 'Sebastian', 'Fernando']

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rostros = rostro.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in rostros:
        # Recortar la cara detectada (en escala de grises)
        face_roi = gray[y:y+h, x:x+w]
        
        # --- ¡Aquí ocurre la magia! ---
        # Predecir a quién pertenece la cara
        # 'label' es el número (0, 1, 2...)
        # 'confidence' es la "confianza" (para LBPH, un valor MÁS BAJO es MEJOR)
        label, confidence = face_recognizer.predict(face_roi)
        
        # --- Mostrar el resultado ---
        
        # Dibujar el rectángulo verde
        cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        
        # Poner el nombre
        # Puedes ajustar el valor de 'confidence' (ej. 70) 
        # para filtrar desconocidos
        if confidence < 89:
            nombre_mostrado = names[label]
            color_texto = (0, 255, 0) # Verde para conocido
        else:
            nombre_mostrado = "Desconocido"
            color_texto = (0, 0, 255) # Rojo para desconocido

        # Mostrar el nombre y la confianza
        texto = f'{nombre_mostrado} ({confidence:.2f})'
        cv.putText(frame, texto, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, color_texto, 2)

    cv.imshow('Reconocimiento Facial', frame)
    
    k = cv.waitKey(1)
    if k == 27: # Presiona ESC para salir
        break

cap.release()
cv.destroyAllWindows()
