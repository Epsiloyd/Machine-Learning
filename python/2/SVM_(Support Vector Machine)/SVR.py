#SVR (Suppor vector regression)

#Importamos librerias
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from matplotlib.ticker import FormatStrFormatter

#Importamos el dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
x = x.reshape(-1,1) 
y = dataset.iloc[:, -1].values
# Redimensionando y (salarios) para aplicar la escala
y = y.reshape(len(y), 1)
print("X (Posiciones):", x)
print("y (Salarios):", y)


#Dividir el dataset en conjunto de entrenamiento y conjunto de testing
"""
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_satet = 0)
"""

#Escalar variables
sc_x = StandardScaler()# Inicializa el escalador para X
sc_y = StandardScaler()# Inicializa el escalador para y
x = sc_x.fit_transform(x)# Escala de posiciones
y = sc_y.fit_transform(y.reshape(-1,1))# Escala de posiciones
y =y.ravel()
print ('x escalado:',x)
print('y escalado',y)

# Entrenamiento del modelo SVR (Regresión con Máquinas de Vectores de Soporte)
regressor = SVR(kernel='rbf')
regressor.fit(x, y)

#Crear aqui nuestro modelo de regresion 

#Prediccion de nuestro modelo con SVR (Support Vector Regression)
resultado  = sc_y.inverse_transform(
    regressor.predict(sc_x.transform(np.array([[6.5]]))).reshape(-1,1))
print('Prediccion del salario para el puesto 6.5:', resultado)

# Visualización de los resultados del modelo SVR
plt.scatter(sc_x.inverse_transform(x), 
            sc_y.inverse_transform(y.reshape(-1,1)),
            color='red')  # Puntos de datos reales
plt.plot(sc_x.inverse_transform(x),
         sc_y.inverse_transform(regressor.predict(x).reshape(-1, 1)),
         color='blue')  # Curva ajustada por SVR
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Nivel de Puesto')
plt.ylabel('Salario')
plt.show()  # Muestra el gráfico

#Visualizacion del resultado del modelo de regresion SVR con mayor resolucion para la curva mas suave
x_grid = np.arange(min(sc_x.inverse_transform(x)),
                   max(sc_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(sc_x.inverse_transform(x),
            sc_y.inverse_transform(y.reshape(-1,1)), color = 'red')
plt.plot(x_grid,sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid)
                                                         ).reshape-1,1), color = 'blue')
plt.title('Truht or Bluff (SVR)')
plt.xlabel('Nivel de puesto')
plt.ylabel('Salario')
plt.show()

# ADICIONAL: Representando el modelo sin que los ejes X e Y estén estandarizados
x_model = np.linspace(x.min(), x.max(), num=100).reshape(-1, 1)  # Rango de valores para predicción del modelo
y_model = regressor.predict(x_model).reshape(-1, 1)  # Predicción del modelo SVR
x_model = sc_x.inverse_transform(x_model)  # Desescalado de X
y_model = sc_y.inverse_transform(y_model)  # Desescalado de y

# Visualización con formato de ejes sin estandarización
plt.figure(figsize=(8, 6))
plt.axes().yaxis.set_major_formatter(FormatStrFormatter('%.0f'))  # Formato de los valores en el eje Y
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='red', label='Datos Reales')  # Datos reales
plt.plot(x_model, y_model, color='green', label='Modelo SVR')  # Modelo ajustado
plt.title('Regresión con SVM (SVR)', fontsize=16)
plt.xlabel('Nivel de Puesto')
plt.ylabel('Salario')
plt.grid()  # Muestra la cuadrícula
plt.legend()  # Muestra la leyenda
plt.show()  # Muestra el gráfico

