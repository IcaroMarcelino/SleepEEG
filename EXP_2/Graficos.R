library(readr)
library(lattice)
############################################################
############################################################
# Mapas de Calor - KNN para cada combinação de pacientes  #
############################################################
############################################################
KNN <- read_csv("~/Downloads/Dados_Sono/GP/SleepEEG/EXP_2/KNN_Analysis/KNN_AVE_K3.csv", col_names = FALSE)
knn <- matrix(nrow = 8, ncol = 8)
for(i in 1:8){
for(j in 1:8){
knn[i,j] <- KNN[i,j][[1]]
}
}
colnames(knn) = c('Ex1', 'Ex2', 'Ex3', 'Ex4', 'Ex5', 'Ex6', 'Ex7', 'Ex8')
rownames(knn) = c('Ex1', 'Ex2', 'Ex3', 'Ex4', 'Ex5', 'Ex6', 'Ex7', 'Ex8')
levelplot(knn, main = 'K = 3', xlab = 'Excerpt for Training', ylab = 'Excerpt for Testing', col.regions = heat.colors(640)[length(heat.colors(640)):1], pretty = TRUE)
KNN <- read_csv("~/Downloads/Dados_Sono/GP/SleepEEG/EXP_2/KNN_Analysis/KNN_AVE_K5.csv", col_names = FALSE)
knn <- matrix(nrow = 8, ncol = 8)
for(i in 1:8){
for(j in 1:8){
knn[i,j] <- KNN[i,j][[1]]
}
}
colnames(knn) = c('Ex1', 'Ex2', 'Ex3', 'Ex4', 'Ex5', 'Ex6', 'Ex7', 'Ex8')
rownames(knn) = c('Ex1', 'Ex2', 'Ex3', 'Ex4', 'Ex5', 'Ex6', 'Ex7', 'Ex8')
levelplot(knn, main = 'K = 5', xlab = 'Excerpt for Training', ylab = 'Excerpt for Testing', col.regions = heat.colors(640)[length(heat.colors(640)):1], pretty = TRUE)
KNN <- read_csv("~/Downloads/Dados_Sono/GP/SleepEEG/EXP_2/KNN_Analysis/KNN_AVE_K7.csv", col_names = FALSE)
knn <- matrix(nrow = 8, ncol = 8)
for(i in 1:8){
for(j in 1:8){
knn[i,j] <- KNN[i,j][[1]]
}
}
colnames(knn) = c('Ex1', 'Ex2', 'Ex3', 'Ex4', 'Ex5', 'Ex6', 'Ex7', 'Ex8')
rownames(knn) = c('Ex1', 'Ex2', 'Ex3', 'Ex4', 'Ex5', 'Ex6', 'Ex7', 'Ex8')
levelplot(knn, main = 'K = 7', xlab = 'Excerpt for Training', ylab = 'Excerpt for Testing', col.regions = heat.colors(640)[length(heat.colors(640)):1], pretty = TRUE)
KNN <- read_csv("~/Downloads/Dados_Sono/GP/SleepEEG/EXP_2/KNN_Analysis/KNN_AVE_K9.csv", col_names = FALSE)
knn <- matrix(nrow = 8, ncol = 8)
for(i in 1:8){
for(j in 1:8){
knn[i,j] <- KNN[i,j][[1]]
}
}
colnames(knn) = c('Ex1', 'Ex2', 'Ex3', 'Ex4', 'Ex5', 'Ex6', 'Ex7', 'Ex8')
rownames(knn) = c('Ex1', 'Ex2', 'Ex3', 'Ex4', 'Ex5', 'Ex6', 'Ex7', 'Ex8')
levelplot(knn, main = 'K = 9', xlab = 'Excerpt for Training', ylab = 'Excerpt for Testing', col.regions = heat.colors(640)[length(heat.colors(640)):1], pretty = TRUE)
KNN <- read_csv("~/Downloads/Dados_Sono/GP/SleepEEG/EXP_2/KNN_Analysis/KNN_AVE_K11.csv", col_names = FALSE)
knn <- matrix(nrow = 8, ncol = 8)
for(i in 1:8){
for(j in 1:8){
knn[i,j] <- KNN[i,j][[1]]
}
}
colnames(knn) = c('Ex1', 'Ex2', 'Ex3', 'Ex4', 'Ex5', 'Ex6', 'Ex7', 'Ex8')
rownames(knn) = c('Ex1', 'Ex2', 'Ex3', 'Ex4', 'Ex5', 'Ex6', 'Ex7', 'Ex8')
levelplot(knn, main = 'K = 11', xlab = 'Excerpt for Training', ylab = 'Excerpt for Testing', col.regions = heat.colors(640)[length(heat.colors(640)):1], pretty = TRUE)
KNN <- read_csv("~/Downloads/Dados_Sono/GP/SleepEEG/EXP_2/KNN_Analysis/KNN_AVE_K13.csv", col_names = FALSE)
knn <- matrix(nrow = 8, ncol = 8)
for(i in 1:8){
for(j in 1:8){
knn[i,j] <- KNN[i,j][[1]]
}
}
colnames(knn) = c('Ex1', 'Ex2', 'Ex3', 'Ex4', 'Ex5', 'Ex6', 'Ex7', 'Ex8')
rownames(knn) = c('Ex1', 'Ex2', 'Ex3', 'Ex4', 'Ex5', 'Ex6', 'Ex7', 'Ex8')
levelplot(knn, main = 'K = 13', xlab = 'Excerpt for Training', ylab = 'Excerpt for Testing', col.regions = heat.colors(640)[length(heat.colors(640)):1], pretty = TRUE)
KNN <- read_csv("~/Downloads/Dados_Sono/GP/SleepEEG/EXP_2/KNN_Analysis/KNN_AVE_K15.csv", col_names = FALSE)
knn <- matrix(nrow = 8, ncol = 8)
for(i in 1:8){
for(j in 1:8){
knn[i,j] <- KNN[i,j][[1]]
}
}
colnames(knn) = c('Ex1', 'Ex2', 'Ex3', 'Ex4', 'Ex5', 'Ex6', 'Ex7', 'Ex8')
rownames(knn) = c('Ex1', 'Ex2', 'Ex3', 'Ex4', 'Ex5', 'Ex6', 'Ex7', 'Ex8')
levelplot(knn, main = 'K = 15', xlab = 'Excerpt for Training', ylab = 'Excerpt for Testing', col.regions = heat.colors(640)[length(heat.colors(640)):1], pretty = TRUE)
KNN <- read_csv("~/Downloads/Dados_Sono/GP/SleepEEG/EXP_2/KNN_Analysis/KNN_AVE_K17.csv", col_names = FALSE)
knn <- matrix(nrow = 8, ncol = 8)
for(i in 1:8){
for(j in 1:8){
knn[i,j] <- KNN[i,j][[1]]
}
}
colnames(knn) = c('Ex1', 'Ex2', 'Ex3', 'Ex4', 'Ex5', 'Ex6', 'Ex7', 'Ex8')
rownames(knn) = c('Ex1', 'Ex2', 'Ex3', 'Ex4', 'Ex5', 'Ex6', 'Ex7', 'Ex8')
levelplot(knn, main = 'K = 17', xlab = 'Excerpt for Training', ylab = 'Excerpt for Testing', col.regions = heat.colors(640)[length(heat.colors(640)):1], pretty = TRUE)
KNN <- read_csv("~/Downloads/Dados_Sono/GP/SleepEEG/EXP_2/KNN_Analysis/KNN_AVE_K19.csv", col_names = FALSE)
knn <- matrix(nrow = 8, ncol = 8)
for(i in 1:8){
for(j in 1:8){
knn[i,j] <- KNN[i,j][[1]]
}
}
colnames(knn) = c('Ex1', 'Ex2', 'Ex3', 'Ex4', 'Ex5', 'Ex6', 'Ex7', 'Ex8')
rownames(knn) = c('Ex1', 'Ex2', 'Ex3', 'Ex4', 'Ex5', 'Ex6', 'Ex7', 'Ex8')
levelplot(knn, main = 'K = 19', xlab = 'Excerpt for Training', ylab = 'Excerpt for Testing', col.regions = heat.colors(640)[length(heat.colors(640)):1], pretty = TRUE)

############################################################
############################################################
# Boxplot - KNN para todos os pacientes                    #
############################################################
############################################################
KNN_AllExcerpt_Test <- read_csv("~/Downloads/Dados_Sono/GP/SleepEEG/EXP_2/KNN_Analysis/KNN_AllExcerpt_Test.csv")
boxplot(KNN_AllExcerpt_Test$Acc~KNN_AllExcerpt_Test$K, xlab = "K", ylab = 'Acurácia', main = 'Classificação com KNN utilizando todos os dados')


############################################################
############################################################
# Boxplot - KNN + GP - Paciente 1                          #
############################################################
############################################################
INFO_GP <- read_csv("~/Downloads/Dados_Sono/GP/SleepEEG/EXP_2/INFO_GP.csv",col_names = FALSE)
boxplot(INFO_GP$X4~INFO_GP$X2, xlab = "K", ylab = 'Acurácia', main = 'KNN + GP\nTodos os pacientes')

############################################################
############################################################
# Boxplot - KNN + Fmeasure                                 #
############################################################
############################################################
k <- read_csv("~/Downloads/Dados_Sono/GP/SleepEEG/EXP_2/KNN_Analysis/KNN_AllExcerpt_Test_Fmeasure.csv")
par(mfrow = c(1,2))
boxplot(k$Precision_S~k$K, main = "KNN - Precision", xlab = "K", col = rgb(0,1,0,.5), border = 'darkgreen', ylim = c(.3,.9))
boxplot(k$Precision_NS~k$K, col = rgb(1,0,0,.5), border = 'darkred', add = TRUE)
legend('bottomright',fill=c("green","red"),legend=c("Spindle","Non-Spindle"), inset=.02, title="Precision measure",bty = "n")

boxplot(k$Recall_S~k$K, main = "KNN - Recall\n", xlab = "K", col = rgb(0,1,0,.5), border = 'darkgreen', ylim = c(0,1))
boxplot(k$Recall_NS~k$K, col = rgb(1,0,0,.5), border = 'darkred', add = TRUE)
legend('right',fill=c("green","red"),legend=c("Spindle","Non-Spindle"), inset=.02, title="Recall measure",bty = "n")

boxplot(k$Fscore_S~k$K, main = "KNN - F-score\n", xlab = "K", col = rgb(0,1,0,.5), border = 'darkgreen', ylim = c(0,1))
boxplot(k$Fscore_NS~k$K, col = rgb(1,0,0,.5), border = 'darkred', add = TRUE)
legend('right',fill=c("green","red"),legend=c("Spindle","Non-Spindle"), inset=.02, title="F-score measure",bty = "n")

boxplot(k$Acc~k$K, main = "KNN - Accuracy\n", xlab = "K", col = rgb(0,0,1,.5), border = 'darkblue', ylim = c(.8,.865))


evolution <- function(name, rang, iter, cl){
  aux <- matrix(nrow = iter, ncol = length(rang))
  est <- c(rep(0,each=iter))
  desvpos <- c(rep(0,each=iter))
  desvneg <- c(rep(0,each=iter))
  desvmax <- c(rep(0,each=iter))
  desvmin <- c(rep(0,each=iter))
  for (i in 1:length(rang)){
    a <- read.csv(paste(name, as.character(rang[i]), ".csv", sep= ""), header = FALSE, quote = "")
    for (j in 1:iter){
      est[j] <- est[j] + as.numeric(a[j,cl]/length(rang))
      aux[j,i] <- as.numeric(a[j,cl])
    }
  }
  for (j in 1:iter){
    desvpos[j] <- sqrt(var(aux[j,]))
    desvneg[j] <- sqrt(var(aux[j,]))
    desvmax[j] <- max(aux[j,])
    desvmin[j] <- min(aux[j,])
  }
  return(list(custos = est, desvneg = est - desvneg, desvpos = desvpos + est, desvmax = desvmax, desvmin = desvmin))
}

par(mfrow= c(1,2))

lev <- unique(k$K)
ds = matrix(nrow = length(lev), ncol = 5)
ds_max = matrix(nrow = length(lev), ncol = 5)
ds_min = matrix(nrow = length(lev), ncol = 5)

ds[,1] <- lev
ds_max[,1] <- lev
ds_min[,1] <- lev

for(i in 1:length(lev)){
  ind <- which(k$K == lev[i])
  vec <- k[ind,c(3,5,7,11)]
  
  for(j in 1:4){
    ds[i,j+1] <- mean(vec[,j][[1]])
    ds_max[i,j+1] <- mean(vec[,j][[1]]) + (sd(vec[,j][[1]]))
    ds_min[i,j+1] <- mean(vec[,j][[1]]) - (sd(vec[,j][[1]]))
  }
}

plot(1,1,type = "l", ylim = c(.1,.9), xlim = c(3,19), main = "KNN - Full dataset", xlab = "K", ylab="", col='white')
polygon(c(ds_max[,1], rev(ds_min[,1])), c(ds_max[,2], rev(ds_min[,2])), col = rgb(1,0,0,.35), border = rgb(1,0,0,.35))
lines(ds[,1],ds[,2], type = "l", col = "red", lwd = 2)

polygon(c(ds_max[,1], rev(ds_min[,1])), c(ds_max[,3], rev(ds_min[,3])), col = rgb(0,1,0,.35), border = rgb(0,1,0,.35))
lines(ds[,1],ds[,3], type = "l", col = "darkgreen", lwd = 2)

polygon(c(ds_max[,1], rev(ds_min[,1])), c(ds_max[,4], rev(ds_min[,4])), col = rgb(0,0,1,.35), border = rgb(0,0,1,.35))
lines(ds[,1],ds[,4], type = "l", col = "blue",lwd = 2)

polygon(c(ds_max[,1], rev(ds_min[,1])), c(ds_max[,5], rev(ds_min[,5])), col = rgb(0,0,0,.35), border = rgb(0,0,0,.35))
lines(ds[,1],ds[,5], type = "l", col = 'black', lwd = 2)


box()