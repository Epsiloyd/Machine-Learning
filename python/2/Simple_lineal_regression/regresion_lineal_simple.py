#regresion lineal simple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importa el data set

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
print("Datos originales de X")
print(x)
y = dataset.iloc[:,1].values
print("Datos originales de Y")
print(y)


#Dividir el data set entre el conjunto de entrenamiento y testing
from sklearn.model_selection import train_test_split
#Usamos en este caso 2/3 para entrenamiento y 1/3 para el test
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=1/3, random_state=0)
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

#Crear modelo de regresion lineal simple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predecir los resultados del conjunto de prueba
y_pred = regressor.predict(x_test)

#Visualizar los resultados de entrenamiento
plt.scatter(x_train, y_train, color='red')#Puntos reales en el conjunto de entrenamiento
plt.plot(x_train,regressor.predict(x_train),color='blue')# Línea de regresión ajustada (misma que en entrenamiento)
plt.title('Salarios vs Experiencia (Conjunto de entrenamiento)')#Titulo grafico
plt.xlabel('Años de experiencia')#Etiqueta eje x
plt.ylabel('Salario')#Etiqueta eje y
plt.show()#Mostrar etiqueta

#Visualizar los resultados de test
plt.scatter(x_test, y_test, color='red')#Puntos reales en el conjunto de test
plt.plot(x_train,regressor.predict(x_train),color='blue')# Línea de regresión ajustada (misma que en entrenamiento)
plt.title('Salarios vs Experiencia (Conjunto de entrenamiento)')#Titulo grafico
plt.xlabel('Años de experiencia')#Etiqueta eje x
plt.ylabel('Salario')#Etiqueta eje y
plt.show()#Mostrar etiqueta