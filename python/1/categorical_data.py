#Plantilla de pre processdos- datos
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

#Codificar datos categoricos
#Codificar datos de la variable dependendiente (x)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
#Aplicamos OneHotEncoder a la primera columna (indice 0)
ct = ColumnTransformer(transformers = [('encoder',OneHotEncoder(),[0])],remainder='passthrough')
x = np.array(ct.fit_transform(x))#Transoformamos x y lo convertimos a un array de np
print('Datos de X de la codificacion de ONEHOT')
print(x)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y = le.fit_transform(y)
print('Datos de y de la codificacion OneHot')
print(y)