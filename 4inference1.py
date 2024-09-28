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

# Inicializar MediaPipe Hands y Face Mesh
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Diccionario de etiquetas
labels_dict = {0: '1', 1: '2', 2: '3', 4: '4'}

# Especificaciones de dibujo
drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)  # Azul, línea delgada, círculos pequeños

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Usar los modelos de MediaPipe Hands y Face Mesh
    hand_results = hands.process(frame_rgb)
    face_results = face_mesh.process(frame_rgb)

    all_landmarks = []
    x_all = []
    y_all = []

    # Dibuja los puntos de ambas manos, si están presentes
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS, 
                drawing_spec, 
                drawing_spec)

            data_aux = []
            x_ = []
            y_ = []

            # Capturar coordenadas de los landmarks de la mano
            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                x_.append(x)
                y_.append(y)

            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

            all_landmarks.append(data_aux)
            x_all.extend(x_)
            y_all.extend(y_)

    # Dibuja los puntos de la cara, si están presentes
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                face_landmarks, 
                mp_face_mesh.FACEMESH_TESSELATION, 
                drawing_spec, 
                drawing_spec)

            face_data_aux = []
            x_face = []
            y_face = []

            for landmark in face_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                x_face.append(x)
                y_face.append(y)
                face_data_aux.append(x)
                face_data_aux.append(y)

            all_landmarks.append(face_data_aux)
            x_all.extend(x_face)
            y_all.extend(y_face)

    # Predicción cuando se detecta al menos una mano o la cara
    if len(all_landmarks) > 0:
        try:
            all_landmarks_flat = [item for sublist in all_landmarks for item in sublist]

            # Realizar la predicción con una o ambas manos y/o la cara
            prediction = model.predict([np.asarray(all_landmarks_flat)])
            predicted_character = labels_dict.get(int(prediction[0]), None)

            if predicted_character:
                # Definir coordenadas para el cuadro alrededor de la mano o cara
                if x_all and y_all:
                    x1 = int(min(x_all) * W) - 10
                    y1 = int(min(y_all) * H) - 10
                    x2 = int(max(x_all) * W) + 10
                    y2 = int(max(y_all) * H) + 10

                    # Cambiar el color del rectángulo (azul brillante)
                    color = (255, 0, 0)  # Azul en formato BGR

                    # Dibujar el cuadro alrededor de las manos o cara
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Mostrar el carácter predicho solo si se detecta una seña
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)
        except Exception as e:
            pass  # Si hay un error, continuamos sin interrumpir el programa

    # Mostrar el video con los landmarks dibujados
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

