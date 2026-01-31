#Regrecion lineal multiple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importa el data set
dataset = pd.read_csv('50_Startups.csv')

# Separar las variables independientes (X) de la variable dependiente (y).
# X: Todas las columnas excepto la última (características).
# y: La última columna (beneficios).

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
print("Datos originales (x):\n",x)

# 2. Codificación de datos categóricos
# La columna 3 (índice 3 en Python) contiene información categórica sobre la localización de la startup.
# Usamos OneHotEncoder para convertir esta columna en variables dummy (codificación one-hot).

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Transformador que aplica OneHotEncoder a la columna de índice 3.
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(drop='first'),[3])],remainder='passthrough')

# Transformamos X y lo convertimos en un array de NumPy.
x =ct.fit_transform(x)
x = x.astype(float)
print("Datos codificados (x):\n",x)

# 3. División del dataset en conjunto de entrenamiento y prueba
# Se divide el conjunto de datos en entrenamiento (80%) y prueba (20%) para evaluar el modelo
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=0)

# 4. Entrenamiento del modelo de regresión lineal múltiple
from sklearn.linear_model import LinearRegression

# Inicializamos el modelo de regresión lineal.
regressor= LinearRegression()

# Ajustamos el modelo a los datos de entrenamiento.
regressor.fit(x_train,y_train)

# 5. Predicción de los resultados del conjunto de prueba
# Usamos el modelo entrenado para predecir los beneficios en el conjunto de prueba
y_pred = regressor.predict(x_test)

# Configuramos la salida de NumPy para mostrar los números con dos decimales.
np.set_printoptions(precision=2)

# Mostramos las predicciones junto con los valores reales para comparación.
# Concatenamos las predicciones (y_pred) y los valores reales (y_test) en columnas.
resultados= np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),axis=1)
print("Predicciones vs Valores Reales:\n",resultados)

#Contruir el modelo optimo del RLM utilizando la Eliminacion hacia atras
import statsmodels.api as sm
x = sm.add_constant(x)
sl = 0.05

x_opt = x[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
print(regressor_OLS.summary())

x_opt = x[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
print(regressor_OLS.summary())

x_opt = x[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
print(regressor_OLS.summary())

x_opt = x[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
print(regressor_OLS.summary())

x_opt = x[:,[0,3]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
print(regressor_OLS.summary())
# Interpretación y visualización 
# Este modelo no tiene visualizaciones específicas, pero se podrían graficar las predicciones
# contra los valores reales para evaluar visualmente el desempeño del modelo.

plt.figure(figsize=(10,6))
plt.scatter(range(len(y_test)),y_test, color='blue',label='Valores reales')
plt.scatter(range(len(y_pred)),y_pred, color = 'red', label = 'Predicciones')
plt.title('Predicciones vs Valores reales')
plt.xlabel('Indice de muestra')
plt.ylabel('Beneficios')
plt.legend()
plt.grid()
plt.show()