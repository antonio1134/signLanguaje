import os
import cv2

# Directorio de datos
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Número de clases y tamaño del dataset
number_of_classes = 3
dataset_size = 100

# Captura de video
cap = cv2.VideoCapture(0)

# Crear directorios para las clases si no existen
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Recolectando datos para la clase {}'.format(j))

    # Esperar hasta que se presione 'q' para comenzar la recolección de datos
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mostrar texto en la pantalla
        cv2.putText(frame, 'Presiona q para iniciar', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        
        # Dibujar un rectángulo guía para asegurar que ambas manos estén en la vista
        height, width, _ = frame.shape
        cv2.rectangle(frame, (width//4, height//4), (3*width//4, 3*height//4), (255, 0, 0), 2)
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Capturar imágenes
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mostrar el frame y dibujar el mismo rectángulo guía
        cv2.rectangle(frame, (width//4, height//4), (3*width//4, 3*height//4), (255, 0, 0), 2)
        cv2.imshow('frame', frame)
        
        # Guardar la imagen
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        
        counter += 1
        cv2.waitKey(25)

# Liberar la captura y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
