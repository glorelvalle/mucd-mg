library(ggplot2)
library(qgraph)
library(gplots)
library(tidyverse)

# Lectura del dataset de Card Sorting facilitado
data <- read.csv(url("http://cardsorting.net/tutorials/25.csv"))

# Selección de las variables (eliminamos "Uniqid", "Startdate", "Starttime", "Endtime", "QID", "Comment")
data <- subset(data, select = -c(1,3,4,5,6, ncol(data)))

# Separamos los datos numéricos de las categorías correspondientes
freqs <- data[,2:ncol(data)]

# Obtenemos la frecuencia por datos únicos
unique_freqs <- data.frame(table(unlist(freqs)))

# Representamos un histograma con los datos correspondientes a las tarjetas
jpeg(file.path("img", paste("histograma", ".png", sep = "")))
ggplot(unique_freqs, aes(x=Var1, y=Freq)) + 
  geom_bar(stat="identity", color="black", fill=c("steelblue4", "darkmagenta"))+
  geom_text(aes(label=Freq), vjust=1.6, color="white", size=4.5)+
  ggtitle("Histograma de los datos numéricos") +
  theme(plot.title = element_text(hjust = 0.5, face="bold")) +
  xlab("Elementos") + ylab("Frecuencia") +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank())
dev.off()

# Calculamos la distancia euclídea entre columnas
distances <- as.matrix(dist(t(freqs), method="euclidean"))

# Mapa de calor correspondiente
jpeg(file.path("img", paste("mapa_calor", ".png", sep = "")))
my_palette <- colorRampPalette(c("darkmagenta", "steelblue4", "lightsteelblue"))
heatmap.2(distances, col=my_palette, dendrogram='row', Rowv=TRUE, Colv=TRUE, trace='none', symkey=FALSE)
title("Mapa de calor de las tarjetas")
dev.off()

# Obtenemos el grafo ponderado de la similud entre tarjetas
jpeg(file.path("img", paste("grafo", ".png", sep = "")))
qgraph(1/(1 + distances), labels=colnames(distances), layout="spring", vsize=7, colors=c("darkmagenta", "steelblue3", "lightsteelblue"), edge.color="plum3")
title("Grafo ponderado de la similitud entre tarjetas")
dev.off()

# Obtenemos las tarjetas más relacionadas
which(min(dist(t(freqs), method="euclidean")) == distances, arr.ind=TRUE)

