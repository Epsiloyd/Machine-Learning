#Bosques alaetorios

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

#Ajustar un modelo con random forest con nuestros conjuntos
set.seed(1234)
regressor = randomForest(x = dataset[-2],
                          y = dataset$Salary,
                          ntree = 500,)

#Prediccion de nuevos resultados con random forest
y_pred = predict(regressor, data.frame(Level = 6.5))

#Visualizacion del modelo de random forest
#Inicializamos ggplot2 para poder graficar
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Regression Model)') +
  xlab('Level') +  
  ylab('Salary')