#Plantilla de pre procsados  -datos faltantes
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

#Tratamientos de los NAs
from sklearn.impute import SimpleImputer
imputer =SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(x[:, 1:3])
x[:,1:3] = imputer.transform(x[:,1:3])
print("Datos de X despues de manejar valores;")
print(x)