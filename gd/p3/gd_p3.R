library(ggplot2)

data <- read.csv(url("http://cardsorting.net/tutorials/25.csv"))
data <- subset(data, select = -c(1,3,4,5,6, ncol(data)))
head(data)

count#ggplot(data) +
#  geom_histogram(aes(x = Carrots, y = ..density..),
#bins = 240, fill = 'olivedrab4', col = 'black')


# hacer histograma sobre los numeros que estan en la matriz para obtener
# la frecuencia de los numeros de la matriz