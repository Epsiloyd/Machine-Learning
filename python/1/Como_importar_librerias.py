#Como importar librerias

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importa el data set

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
print("Datos originales de X")
print(x)
y = dataset.iloc[:,3].values
print("Datos originales de Y")
print(y)


#Dividir el data set entre el conjunto de entrenamiento y testing
from sklearn.model_selection import train_test_split
#Usamos el 80% para tratamiento y 20% para prueba
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)
print("Conjunto de entrenamiento (x_train):")
print(x_train)
print("Conjunto de prueba (x_test):")
print(x_test)
print("Etiquetas de conjunto de entrenamiento (y_train):")
print(y_train)
print("Etiqueta de conjunto de prueba (y_test):")
print(y_test)

#Escalado de variables
"""from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:,:3] = sc.fit_transform(x_train[:,:3])#Ajustamos y transformamos las caracteristicas numericas del conjuno del conjunto de entrenamiento
x_test[:,:3] = sc.transform(x_test[:,:3])#Transformamos el conjunto de preba usando el ajuste del conjunto de entrenamiento
print("Conjunto de entrenamiento despues de escalado de caracteristicas:")
print(x_train)
print("Conjunto de pruebas despues del escalado de caracteisiticas:")
print(x_test)"""