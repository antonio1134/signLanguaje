import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

# Cargar datos desde el archivo pickle
try:
    with open('./data.pickle', 'rb') as f:
        data_dict = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError("El archivo 'data.pickle' no se encontró. Asegúrate de que el archivo existe y la ruta es correcta.")
except pickle.UnpicklingError:
    raise ValueError("Error al deserializar el archivo 'data.pickle'. Asegúrate de que el archivo está en el formato correcto.")

# Asegurarse de que los datos se han cargado correctamente
data = np.asarray(data_dict.get('data', []))
labels = np.asarray(data_dict.get('labels', []))

if data.size == 0 or labels.size == 0:
    raise ValueError("Los datos o las etiquetas están vacíos. Asegúrate de que 'data.pickle' contiene datos válidos.")

# Verificar la distribución de clases
class_counts = Counter(labels)
print("Distribución de clases:", class_counts)

# Asegurarse de que cada clase tiene al menos dos ejemplos
for label, count in class_counts.items():
    if count < 2:
        raise ValueError(f"La clase '{label}' tiene menos de dos ejemplos. Asegúrate de que cada clase tenga al menos dos ejemplos.")

# Dividir los datos en conjuntos de entrenamiento y prueba
try:
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
except ValueError as e:
    raise ValueError(f"Error al dividir los datos en conjuntos de entrenamiento y prueba: {e}")

# Inicializar el modelo de Random Forest
model = RandomForestClassifier()

# Entrenar el modelo
model.fit(x_train, y_train)

# Realizar predicciones sobre el conjunto de prueba
y_predict = model.predict(x_test)

# Calcular la precisión del modelo
score = accuracy_score(y_predict, y_test)

print(f'{score * 100:.2f}% de las muestras fueron clasificadas correctamente!')

# Guardar el modelo entrenado en un archivo
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
