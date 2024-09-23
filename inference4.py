import pickle
import cv2
import mediapipe as mp
import numpy as np

# Cargar el modelo
with open('./model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# Inicializar la captura de video
cap = cv2.VideoCapture(0)

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Diccionario de etiquetas
labels_dict = {0: 'casa', 1: 'B', 2: 'C', 4: 'A'}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Usar el modelo de landmarks de MediaPipe Hands
    results = hands.process(frame_rgb)
    all_landmarks = []
    x_all = []
    y_all = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # imagen en la que dibujar
                hand_landmarks,  # salida del modelo
                mp_hands.HAND_CONNECTIONS,  # conexiones de la mano
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            data_aux = []
            x_ = []
            y_ = []

            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                x_.append(x)
                y_.append(y)

            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            all_landmarks.append(data_aux)
            x_all.extend(x_)
            y_all.extend(y_)

        if len(all_landmarks) == 1:
            # Si solo hay una mano, duplicar las características de la mano presente
            all_landmarks.append(all_landmarks[0])
        
        if len(all_landmarks) == 2:
            # Aplanar la lista de características de ambas manos
            all_landmarks_flat = [item for sublist in all_landmarks for item in sublist]
            prediction = model.predict([np.asarray(all_landmarks_flat)])
            predicted_character = labels_dict[int(prediction[0])]

            x1 = int(min(x_all) * W) - 10
            y1 = int(min(y_all) * H) - 10
            x2 = int(max(x_all) * W) + 10
            y2 = int(max(y_all) * H) + 10

            # Cambiar el color del rectángulo (en este caso, verde brillante)
            color = (0, 255, 0)  # Verde en formato BGR

            # Dibujar el cuadro alrededor de las manos
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
