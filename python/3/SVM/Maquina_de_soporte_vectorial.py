

#Support Vector Machine (SVM)

#Importamos librerias
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

#Importamos el dataset
dataset = pd.read_csv('Social_Network_ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Dividir el dataset en conjunto de entrenamiento y conjunto de testing
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 0)
print("Conjunto de entrenamiento (X_train):")
print(x_train)
print("Etiquetas de entrenamiento (y_train):")
print(y_train)
print("Conjunto de prueba (X_test):")
print(x_test)
print("Etiquetas de prueba (y_test):")
print(y_test)

#Escalar variables
sc = StandardScaler()
x_train = sc.fit_transform(x_train)# Ajuste y transformación para el conjunto de entrenamiento
x_test = sc.transform(x_test) # Transformación para el conjunto de prueba
print("Conjunto de entrenamiento escalado (X_train):")
print(x_train)
print("Conjunto de prueba escalado (X_test):")
print(x_test)


#Ajustar el SVM en el conjunto de entrenamiento
classifier = SVC(kernel= 'linear', random_state = 0)
classifier.fit(x_train, y_train)

#Prediccion del conjunto de prueba
y_pred = classifier.predict(x_test)
print("Prediccion sobre el conjunto de prueba")
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#Creando la matriz de confucion
cm = confusion_matrix(y_test, y_pred) #Genera la matriz de confucion
print("Matriz de confucion")
print(cm)
print("Precision del modelo:")
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# Visualización de los resultados en el conjunto de entrenamiento
x_set, y_set = sc.inverse_transform(x_train), y_train  # Inversa para visualizar en la escala original
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 10, stop = x_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = x_set[:, 1].min() - 1000, stop = x_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(x1, x2, classifier.predict(sc.transform(np.array([x1.ravel(), x2.ravel()]).T)).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Conjunto de Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Salario Estimado')
plt.legend()
plt.show()

# Visualización de los resultados en el conjunto de prueba
x_set, y_set = sc.inverse_transform(x_test), y_test  # Inversa para visualizar en la escala original
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 10, stop = x_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = x_set[:, 1].min() - 1000, stop = x_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(x1, x2, classifier.predict(sc.transform(np.array([x1.ravel(), x2.ravel()]).T)).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Conjunto de Prueba)')
plt.xlabel('Edad')
plt.ylabel('Salario Estimado')
plt.legend()
plt.show()