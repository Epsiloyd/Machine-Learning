# -*- coding: utf-8 -*-
#REGRESION CON ARBOL DE DECISION

#Plantilla de regresion

#Importamos librerias
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor

#Importamos el dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#Entrenamiento de modelo de regesion con arbol de decision
regressor = DecisionTreeRegressor(random_state=0)
#Inicializamos el modelo con una semilla aleatoria para reproducinilidad
regressor.fit(x, y)
#Ajustamos el arbol de deciciones a los datos

#Prediccion de un nuevo resultado (para un puesto de nivel 6.5)
resultado = regressor.predict([[6.5]])
print(f'Prediccion del salario del puesto 6.5: {resultado}')

#Visualizacion del resultado de la regresion con el arbol de decision
x_grid = np.arange(min(x),max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid,regressor.predict(x_grid), color = 'blue')
plt.title('Truht or Bluff (Regresion Con Arbol de decision)')
plt.xlabel('Nivel de puesto')
plt.ylabel('Salario')
plt.show()
