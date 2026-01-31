#platilla para pre porcesado de datos

#importar el data set 

dataset = read.csv('Position_Salaries.csv')
dataset=dataset[, 2:3]

#Dividir los datos en conjunto de entrenamiento y conjunto de test
#install.packages("caTools")
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Purchased,SplitRatio = 0.8)
# training_set = subset(dataset,split==TRUE)
# testing_set = subset(dataset,split==FALSE)

#Escalado de valores
# training_set[,2:3]=scale(training_set[,2:3])
# testing_set[,2:3]=scale(testing_set[,2:3])

#Ajustar modelo de regresion lineal con el conjunto de datos 
lin_reg = lm(formula = Salary ~ .,
             data = dataset)


#Ajustar un modelo con regresion polinomica con nuestros conjuntos
dataset$Level2 = dataset$Level^2#Llamamos al dataset y creamos una nueva columna de salary y a esta misma la potenciamos al 2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ ., 
              data = dataset)

#Visualizacion del modelo lineal
#Inicializamos ggplot2 para poder graficar
ggplot() +
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             color = 'red') + 
  geom_line(aes( x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
            color = 'blue') +
  ggtitle('Prediccion lineal del sueldo en funcion del nivel del empleado') +
  xlab('Nivel del empleado') + 
  ylab('Sueldo (en $)')



#Visualizacion del modelo polinomico 
x_grid = seq(min(dataset$Level), max(dataset$Level),0.1)
ggplot() +
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             color = 'red') + 
  geom_line(aes( x = x_grid, y = predict(poly_reg, 
                                         newdata = data.frame(Level = x_grid,
                                                              Level2 = x_grid^2,
                                                              Level3 = x_grid^3,
                                                              Level4 = x_grid^4))),
            color = 'blue') +
  ggtitle('Prediccion polinomica del sueldo en funcion del nivel del empleado') +
  xlab('Nivel del empleado') + 
  ylab('Sueldo (en $)')

#Prediccion de nuevos resultados con regrecion lineal 
y_pred = predict(lin_reg, newdata = data.frame(Level = 6.5))

#Prediccion de nuevos resultados con regresion polinomica
y_pred_poly = predict(poly_reg, newdata = data.frame(Level = 6.5,
                                                 Level2 = 6.5^2 ,
                                                 Level3 = 6.5^3,
                                                 Level4 = 6.5^4))
#En la prediccion polinomica fue necesario en mi caso agregar las potencias de cada level para poder obtener un resultado optimo

