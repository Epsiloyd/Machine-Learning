# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 21:38:45 2026

@author: PC BULLOCK
"""

# Regresion con Bosques aleatorios

#Importamos librerias
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
#from sklearn.model_selection import train_test_split

#Importamos el dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#Dividir el dataset en conjunto de entrenamiento y conjunto de testing
"""
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_satet = 0)
"""

#Escalar variables
"""
sc_x = StandarScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
"""

#Ajustar la regresion con el random forest
#Crear aqui nuestro modelo de regresion 
regressor = RandomForestRegressor(n_estimators= 100, random_state=0)
regressor.fit(x, y)

#Prediccion de nuestro modelo
resultado = regressor.predict([[6.5]])
print(f"Prediccion del salario para el puesto 6.5: {resultado}")

#Visualizacion del resultado del modelo random forest

x_grid = np.arange(min(x),max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid,regressor.predict(x_grid), color = 'blue')
plt.title('Truht or Bluff (Regresion Con Random forest)')
plt.xlabel('Nivel de puesto')
plt.ylabel('Salario')
plt.show()
