#Regresion Polinomica
# Plantilla de preprocesamiento de datos

#Cambios realizados:
#Verificación de valores nulos: Se añade un chequeo para detectar si el dataset tiene valores nulos y advertir al usuario.
#Comentarios detallados: Cada sección incluye explicaciones claras sobre qué hace el código.
#Mensajes de depuración: Se añaden prints para verificar las dimensiones de los conjuntos de entrenamiento y prueba.

# Importando las bibliotecas necesarias
import numpy as np  # Para manejo de vectores y matrices
import matplotlib.pyplot as plt  # Para visualización de datos
import pandas as pd  # Para manipulación y análisis de datos
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Importar el dataset
# Asegúrate de que 'Data.csv' esté en el mismo directorio o proporciona la ruta completa.
dataset = pd.read_csv('Position_Salaries.csv')

# Variables independientes (X) y variable objetivo (y)
# Seleccionamos todas las columnas excepto la última como variables independientes (X)
x = dataset.iloc[:, 1:-1].values
x = x.reshape(-1,1)

# Seleccionamos la última columna como la variable objetivo (y)
y = dataset.iloc[:, -1].values

#Entrenamiento del modelo de regrecion linal con todo el data set
lin_reg = LinearRegression()
lin_reg.fit(x, y)

#Entrenamiento del modelo de regresion polinomica
#Se utiliza un grado de 4 para el polinomio
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)
y_pred = lin_reg_2.predict(x_poly)

#Visualizacion de los resultado de la regresion lineal
plt.scatter(x,y, color = 'red')
plt.plot(x,lin_reg.predict(x), color = 'blue')
plt.title('Truth of Bluff (Regresion lineal)')
plt.xlabel('Nivel de puesto')
plt.ylabel('Salario')
plt.show()

#Visualizacion de los resultado en la regresion polinomica
plt.scatter(x, y, color='red')
plt.plot(x,lin_reg_2.predict(x_poly), color='blue')
plt.title('Truth or Bluff (Regresion polinomica)')
plt.xlabel('Nivel de puesto')
plt.ylabel('Salario')
plt.show()

# Visualización de los resultados de la Regresión Polinómica con mayor resolución para una curva más suave
x_grid = np.arange(min(x),max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid,lin_reg_2.predict(poly_reg.transform(x_grid)), color = 'blue')
plt.title('Truht or Bluff (Regresion polinomica)')
plt.xlabel('Nivel de puesto')
plt.ylabel('Salario')
plt.show()

#Prediccion del modelo de regrecion lineal
salario_lineal = lin_reg.predict([[6.5]])
print(f"Prediccion con regrsion lineal para el puesto 6.5: {salario_lineal}")

#Prediccion con el modelo de regresion lineal polinomica
salario_polinomica = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(f"Prediccion con regresion polinomica para el puesto 6.5:{salario_polinomica}")