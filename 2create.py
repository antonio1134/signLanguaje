import os
import pickle
import numpy as np

DATA_DIR = './data'

data = []
labels = []

# Iterar sobre las clases en el directorio de datos
for dir_ in os.listdir(DATA_DIR):
    # Verificar si el elemento es un directorio (una clase)
    if not os.path.isdir(os.path.join(DATA_DIR, dir_)):
        continue

    # Iterar sobre los archivos de datos en cada clase
    for file_name in os.listdir(os.path.join(DATA_DIR, dir_)):
        if file_name.endswith('.npy'):
            # Cargar los datos de puntos clave desde el archivo .npy
            file_path = os.path.join(DATA_DIR, dir_, file_name)
            keypoints = np.load(file_path)

            # Añadir los datos y las etiquetas (clase correspondiente)
            data.append(keypoints)
            labels.append(dir_)

# Guardar los datos y las etiquetas en un archivo pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
