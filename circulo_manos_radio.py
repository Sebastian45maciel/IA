import cv2
import mediapipe as mp
import numpy as np

# --- 1. Inicialización de MediaPipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# max_num_hands=1 es suficiente si solo usamos una mano para el control
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1) 

# --- 2. Parámetros del Dibujo ---
CIRCULO_COLOR = (0, 255, 0)  # Verde
TEXTO_COLOR = (255, 255, 255) # Blanco
MIN_PINCH_THRESHOLD = 20 # Distancia mínima para que el círculo empiece a dibujarse

# --- 3. Captura de Video y Bucle Principal ---
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) # Voltear para control intuitivo
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Variables para el centro y radio del círculo
    center_point = None
    radius = 0
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            
            label = handedness.classification[0].label # 'Left' o 'Right'
            
            # SOLO procesamos la mano derecha para el control
            if label == 'Right': 
                
                # Obtener coordenadas de los puntos clave (índice 8 y pulgar 4)
                p_index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                p_thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                
                # Convertir a coordenadas de píxeles
                idx_tip = (int(p_index.x * w), int(p_index.y * h))
                thumb_tip = (int(p_thumb.x * w), int(p_thumb.y * h))
                
                # 1. Calcular la distancia (radio)
                pinch_distance = np.linalg.norm(np.array(idx_tip) - np.array(thumb_tip))

                if pinch_distance > MIN_PINCH_THRESHOLD:
                    
                    # 2. Definir el radio
                    # Usamos la distancia como el radio, con un límite máximo
                    radius = int(min(pinch_distance, 300)) # Limite el radio a 300px
                    
                    # 3. Definir el centro del círculo
                    # Usamos el punto medio entre los dos dedos como centro
                    center_x = (idx_tip[0] + thumb_tip[0]) // 2
                    center_y = (idx_tip[1] + thumb_tip[1]) // 2
                    center_point = (center_x, center_y)
                    
                    # Opcional: Dibujar las puntas de los dedos y la línea de radio
                    cv2.circle(frame, idx_tip, 8, (255, 0, 0), -1)
                    cv2.circle(frame, thumb_tip, 8, (0, 0, 255), -1)
                    cv2.line(frame, idx_tip, thumb_tip, (100, 100, 100), 2)


    # --- 4. Dibujar el Círculo y la Información ---
    
    if center_point is not None and radius > 0:
        # Dibujar el círculo
        cv2.circle(frame, center_point, radius, CIRCULO_COLOR, 3)
        cv2.circle(frame, center_point, 5, CIRCULO_COLOR, -1) # Dibujar el centro

        # Mostrar el valor del radio
        text = f"Radio: {radius} px"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, TEXTO_COLOR, 2, cv2.LINE_AA)
        cv2.putText(frame, "Mano: Derecha", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, TEXTO_COLOR, 2, cv2.LINE_AA)


    # --- 5. Mostrar y Salir ---
    cv2.imshow("Control de Circulo con Pinza", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
