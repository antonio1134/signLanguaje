import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Cargar los datos recolectados en la fase 2
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = data_dict['data']
labels = np.asarray(data_dict['labels'])

# Determinar la longitud máxima de los datos de puntos clave
max_length = max([len(d) for d in data])

# Rellenar todas las secuencias de puntos clave con ceros hasta que todas tengan la misma longitud
padded_data = [np.pad(d, (0, max_length - len(d)), 'constant') for d in data]
padded_data = np.asarray(padded_data)

# Dividir los datos en conjunto de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(padded_data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Inicializar el modelo RandomForestClassifier
model = RandomForestClassifier()

# Entrenar el modelo con los datos de entrenamiento
model.fit(x_train, y_train)

# Realizar predicciones sobre los datos de prueba
y_predict = model.predict(x_test)

# Calcular la precisión del modelo
score = accuracy_score(y_predict, y_test)

print('{}% de las muestras fueron clasificadas correctamente!'.format(score * 100))

# Guardar el modelo entrenado en un archivo pickle
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
