import os
import cv2
import mediapipe as mp
import numpy as np

# Directorio para almacenar los datos
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3  # Número de clases
dataset_size = 120     # Tamaño del dataset

# Inicializar MediaPipe Hands y Face Mesh
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Inicializar la captura de video
cap = cv2.VideoCapture(0)

# Recolectar datos para cada clase
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Recolectando datos para la clase {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Presiona q para iniciar la recolección', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Usar los modelos de MediaPipe Hands y Face Mesh
        hand_results = hands.process(frame_rgb)
        face_results = face_mesh.process(frame_rgb)

        data_aux = []
        x_all = []
        y_all = []

        # Capturar puntos clave de las manos
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                x_ = []
                y_ = []

                for landmark in hand_landmarks.landmark:
                    x = landmark.x
                    y = landmark.y
                    x_.append(x)
                    y_.append(y)

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

                x_all.extend(x_)
                y_all.extend(y_)

        # Capturar puntos clave de la cara
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    data_aux.append(landmark.x)
                    data_aux.append(landmark.y)

        # Guardar la información si se detectan puntos clave
        if len(data_aux) > 0:
            np.save(os.path.join(class_dir, '{}.npy'.format(counter)), np.array(data_aux))
            counter += 1

        # Mostrar el video con los puntos clave dibujados
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
