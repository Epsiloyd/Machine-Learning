#Regresion lineal
dataset = read.csv('Salary_Data.csv')
# dataset=dataset[,2:3]

#Dividir los datos en conjunto de entrenamiento y conjunto de test
#install.packages("caTools")
library(caTools)
set.seed(123)

#Escalado de valores
# training_set[,2:3]=scale(training_set[,2:3])
# testing_set[,2:3]=scale(testing_set[,2:3])
split = sample.split(dataset$Salary,SplitRatio = 2/3)
training_set = subset(dataset,split==TRUE)
testing_set = subset(dataset,split==FALSE)

#Ajustar el modelo de regresion lineal con el conjunto de entenamiento
regressor = lm(formula =Salary ~ YearsExperience,
               data = training_set)

#prediccion de resultados con el conjutno de test
y_pred = predict(regressor,newdata = testing_set)

#Visualizacion de los resultados en el conjunto de entrenamiento
#install.packages("ggplot2")
ggplot() + 
  geom_point(aes(x=training_set $YearsExperience,y = training_set$Salary),
             colour = "red") +
  geom_line(aes(x = training_set$YearsExperience, 
                y = predict(regressor,newdata = training_set)),
            colour = "blue")+
  ggtitle("Sueldo vs Años de experiencia(Conjunto de entrenamiento)")+
  xlab("Años de experiencia")+
  ylab("Sueldo en $")

#Visualizacion de los resultados en el conjunto de testing
#install.packages("ggplot2")
ggplot() + 
  geom_point(aes(x=testing_set $YearsExperience,y = testing_set$Salary),
             colour = "red") +
  geom_line(aes(x = training_set$YearsExperience, 
                y = predict(regressor,newdata = training_set)),
            colour = "blue")+
  ggtitle("Sueldo vs Años de experiencia(Conjunto de testing)")+
  xlab("Años de experiencia")+
  ylab("Sueldo en $")