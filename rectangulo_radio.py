# python 12
# rectangulo que rota y es con las manos
import cv2
import mediapipe as mp
import numpy as np

# --- 1. Inicialización de MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1) 

# --- 2. Parámetros del Dibujo ---
RECTANGULO_COLOR = (0, 0, 255)  # Rojo
TEXTO_COLOR = (255, 255, 255) # Blanco
MIN_DISTANCE_THRESHOLD = 30 # Distancia mínima para que el rectángulo empiece a dibujarse
RECT_WIDTH_FACTOR = 0.5 # Controla el ancho del rectángulo en relación a la longitud de la diagonal

# --- 3. Funciones de Ayuda ---

# Función para rotar un punto (necesaria para dibujar el rectángulo rotado)
def rotate_point(p, center, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    x = p[0] - center[0]
    y = p[1] - center[1]
    x_new = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    y_new = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    return int(x_new + center[0]), int(y_new + center[1])

# Función para obtener los 4 puntos de un rectángulo rotado,
# definido por su centro, largo, ancho y ángulo.
def get_rotated_rect_points(center, length, width, angle_deg):
    half_l = length / 2
    half_w = width / 2
    
    # Puntos del rectángulo no rotado (centrado en 0,0)
    points_unrotated = [
        (-half_l, -half_w),
        ( half_l, -half_w),
        ( half_l,  half_w),
        (-half_l,  half_w)
    ]
    
    rotated_points = []
    # Rotar cada punto alrededor del centro (0,0) y luego trasladarlo al centro real
    for p_x, p_y in points_unrotated:
        rotated_point = rotate_point((p_x, p_y), (0,0), angle_deg)
        rotated_points.append((int(rotated_point[0] + center[0]), int(rotated_point[1] + center[1])))
        
    return rotated_points

# --- 4. Captura de Video y Bucle Principal ---
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Parámetros del rectángulo a calcular
    rect_center = None
    rect_length = 0
    rect_width = 0
    rect_angle = 0
    
    point1 = None # Pulgar
    point2 = None # Índice
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            
            label = handedness.classification[0].label
            
            if label == 'Right': 
                
                # Obtener coordenadas de los puntos clave (pulgar 4 e índice 8)
                p_thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                p_index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # Convertir a coordenadas de píxeles
                point1 = (int(p_thumb.x * w), int(p_thumb.y * h))
                point2 = (int(p_index.x * w), int(p_index.y * h))
                
                # Calcular la distancia entre los dedos
                distance = np.linalg.norm(np.array(point1) - np.array(point2))

                if distance > MIN_DISTANCE_THRESHOLD:
                    
                    # 1. POSICIÓN (Centro)
                    # El centro es el punto medio entre los dos dedos
                    center_x = (point1[0] + point2[0]) // 2
                    center_y = (point1[1] + point2[1]) // 2
                    rect_center = (center_x, center_y)

                    # 2. TAMAÑO (Largo)
                    # El largo del rectángulo es la distancia entre los dedos
                    rect_length = int(distance)
                    rect_width = int(rect_length * RECT_WIDTH_FACTOR) # Ancho es un factor del largo

                    # 3. ROTACIÓN (Ángulo)
                    # El ángulo está definido por la línea que conecta los dos dedos
                    angle_rad = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])
                    rect_angle = np.degrees(angle_rad)
                    
                    # Opcional: Dibujar los "nodos" de control en la punta de los dedos
                    cv2.circle(frame, point1, 8, (255, 0, 0), -1) # Pulgar
                    cv2.circle(frame, point2, 8, (0, 255, 0), -1) # Índice


    # --- 5. Dibujar el Rectángulo y la Información ---
    
    if rect_center is not None and rect_length > 0 and rect_width > 0:
        
        # Obtener los 4 puntos del rectángulo rotado
        points = get_rotated_rect_points(rect_center, rect_length, rect_width, rect_angle)
        
        # Convertir los puntos a un formato para cv2.polylines
        points_array = np.array(points, np.int32).reshape((-1, 1, 2))
        
        # Dibujar el rectángulo
        cv2.polylines(frame, [points_array], True, RECTANGULO_COLOR, 3)
        
        # Mostrar la información
        text_size = f"Largo: {rect_length}px, Ancho: {rect_width}px"
        text_angle = f"Angulo: {int(rect_angle)} deg"
        cv2.putText(frame, text_size, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXTO_COLOR, 2, cv2.LINE_AA)
        cv2.putText(frame, text_angle, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXTO_COLOR, 2, cv2.LINE_AA)
        cv2.putText(frame, "Control: Pinza Rotatoria", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXTO_COLOR, 2, cv2.LINE_AA)


    # --- 6. Mostrar y Salir ---
    cv2.imshow("Control de Rectangulo Rotatorio", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
