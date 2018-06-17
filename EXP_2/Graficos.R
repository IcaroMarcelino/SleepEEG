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
  require(readr)
  aux <- matrix(nrow = iter, ncol = length(rang))
  est <- c(rep(0,each=iter))
  desvpos <- c(rep(0,each=iter))
  desvneg <- c(rep(0,each=iter))
  desvmax <- c(rep(0,each=iter))
  desvmin <- c(rep(0,each=iter))
  for (i in rang){
    a <- read_table2(paste(name, i, ".csv", sep = ''), col_names = FALSE, skip = 3)
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
sel_best <- function(name, rang, iter, cl){
  aux <- matrix(nrow = iter, ncol = length(rang))
  for (i in rang){
    a <- read_table2(paste(name, i, ".csv", sep = ''), col_names = FALSE, skip = 3)
    for (j in 1:iter){
      aux[j,i] <- as.numeric(a[j,cl])
    }
  }
  aux1 <- matrix(nrow = length(rang), ncol = 1)
  for (j in 1:length(rang)){
    aux1[j] = aux[iter,j]
  }
  print(paste(name, as.character(rang[which(aux1 == max(aux1))]), sep = ""))
  return(read_table2(paste(name, as.character(rang[which(aux1 == max(aux1))[1]]), ".csv", sep = ''), col_names = FALSE, skip = 3))
}

k_ind = 3
k <- evolution(paste("Log_Exec/LOG_GP_EEG_K", k_ind, "__", sep = "" ), c(1:11), 300, 4)
b <- sel_best(paste("Log_Exec/LOG_GP_EEG_K", k_ind, "__", sep = "" ), c(1:11), 300, 4)

plot(k$custos, type = 'l', ylim = c(.75,.95), lwd = 2, col = 'red')
lines(k$custos, type = 'l', ylim = c(.75,.95), lwd = 2, col = 'red')
polygon(c(c(1:300), rev(c(1:300))), c(k$desvpos, rev(k$desvneg)), col = rgb(1,0,0,.35), border = rgb(1,0,0,.35))
lines(b$X4, lty = 2, col = 'red')

k_ind = 3
k <- evolution(paste("Log_Exec/LOG_GP_EEG_K", k_ind, "__", sep = "" ), c(16:30), 300, 4)
plot(k$custos, type = 'l', ylim = c(.8,.91), lwd = 2, col = 'red')

k_ind = 5
k <- evolution(paste("Log_Exec/LOG_GP_EEG_K", k_ind, "__", sep = "" ), c(1:11), 300, 4)
lines(k$custos, type = 'l', ylim = c(.75,.95), lwd = 2, col = 'red')

k_ind = 7
k <- evolution(paste("Log_Exec/LOG_GP_EEG_K", k_ind, "__", sep = "" ), c(1:11), 300, 4)
lines(k$custos, type = 'l', ylim = c(.75,.95), lwd = 2, col = 'red')

k_ind = 9
k <- evolution(paste("Log_Exec/LOG_GP_EEG_K", k_ind, "__", sep = "" ), c(1:11), 300, 4)
lines(k$custos, type = 'l', ylim = c(.75,.95), lwd = 2, col = 'red')

k_ind = 11
k <- evolution(paste("Log_Exec/LOG_GP_EEG_K", k_ind, "__", sep = "" ), c(1:11), 300, 4)
lines(k$custos, type = 'l', ylim = c(.75,.95), lwd = 2, col = 'red')

k_ind = 13
k <- evolution(paste("Log_Exec/LOG_GP_EEG_K", k_ind, "__", sep = "" ), c(1:11), 300, 4)
lines(k$custos, type = 'l', ylim = c(.75,.95), lwd = 2, col = 'red')

k_ind = 15
k <- evolution(paste("Log_Exec/LOG_GP_EEG_K", k_ind, "__", sep = "" ), c(1:11), 300, 4)
lines(k$custos, type = 'l', ylim = c(.75,.95), lwd = 2, col = 'red')

k_ind = 17
k <- evolution(paste("Log_Exec/LOG_GP_EEG_K", k_ind, "__", sep = "" ), c(1:11), 300, 4)
lines(k$custos, type = 'l', ylim = c(.75,.95), lwd = 2, col = 'red')

k_ind = 19
k <- evolution(paste("Log_Exec/LOG_GP_EEG_K", k_ind, "__", sep = "" ), c(1:11), 300, 4)
lines(k$custos, type = 'l', ylim = c(.75,.95), lwd = 2, col = 'red')



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


###################################################################################

draw_ellipses_ACC <- function(infoGP, k_ind, pch, k){
  require(plotrix)
  x = mean(infoGP$Acc[k_ind])
  y = mean(infoGP$Acc[k_ind])
  a = sd(infoGP$Acc[k_ind])
  b = sd(infoGP$Acc[k_ind])
  draw.ellipse(x, y, a, b, col = rgb(0,0,0,.35),border = rgb(0,0,0,.35))
  points(x, y, pch = pch, col = 'black')
  #text(x,y,k,pos=4)
}

draw_ellipses_ROC <- function(infoGP, k_ind, pch, k){
  require(plotrix)
  x = mean(infoGP$TPR_S[k_ind])
  y = mean(infoGP$PPV_S[k_ind])
  a = sd(infoGP$TPR_S[k_ind])
  b = sd(infoGP$PPV_S[k_ind])
  draw.ellipse(x, y, a, b, col = rgb(0,1,0,.35),border = rgb(0,1,0,.35))
  points(x, y, pch = pch, col = 'darkgreen')
  text(x,y,k,pos=2)
  
  x = mean(infoGP$TPR_NS[k_ind])
  y = mean(infoGP$PPV_NS[k_ind])
  a = sd(infoGP$TPR_NS[k_ind])
  b = sd(infoGP$PPV_NS[k_ind])
  draw.ellipse(x, y, a, b, col = rgb(1,0,0,.35),border = rgb(1,0,0,.35))
  points(x, y, pch = pch, col = 'red')
  text(x,y,k,pos=2)
}

library(readr)
INFO_GP <- read_csv("~/Downloads/Dados_Sono/GP/SleepEEG/EXP_3/INFO_GP.csv", col_names = FALSE)
colnames(INFO_GP) <- c("DEEP MAX","K","#Exec","Acc","PPV_S","PPV_NS","TPR_S","TPR_NS","F1_S","F1_NS","SUP_S","SUP_NS","Deep","Training Time")

INFO_GP <- read_csv("~/Downloads/Dados_Sono/GP/SleepEEG/EXP_2/INFO_GP.csv", col_names = FALSE)
colnames(INFO_GP) <- c("DEEP MAX","K","#Exec","b","bb","PPV_S","PPV_NS","TPR_S","TPR_NS","F1_S","F1_NS","SUP_S","SUP_NS","Deep","Training Time")

INFO_GP <- read_csv("~/Downloads/Dados_Sono/GP/SleepEEG/EXP_4/INFO_GP.csv", col_names = FALSE)
colnames(INFO_GP) <- c("DEEP MAX","K","#Exec","Acc","PPV_S","PPV_NS","TPR_S","TPR_NS","F1_S","F1_NS","SUP_S","SUP_NS","Deep","Training Time")

INFO_GP <- read_csv("~/Downloads/Dados_Sono/GP/SleepEEG/EXP_5/INFO_GP.csv", col_names = FALSE)
colnames(INFO_GP) <- c("DEEP MAX","K","#Exec","Acc","PPV_S","PPV_NS","TPR_S","TPR_NS","F1_S","F1_NS","SUP_S","SUP_NS","Deep","Training Time")

INFO_GP <- read_csv("~/Downloads/Dados_Sono/GP/SleepEEG/EXP_6/INFO_GP.csv", col_names = FALSE)
colnames(INFO_GP) <- c("DEEP MAX","K","#Exec","Acc","PPV_S","PPV_NS","TPR_S","TPR_NS","F1_S","F1_NS","SUP_S","SUP_NS","Deep","Training Time")

INFO_GP <- read_csv("~/Downloads/Dados_Sono/GP/SleepEEG/Exp/Try_1/infoGP.csv", col_names = TRUE)
INFO_GP <- read_csv("~/Downloads/Dados_Sono/GP/SleepEEG/Exp/Try_2/infoGP.csv", col_names = TRUE)

k3 <- which(INFO_GP$K == 3)
k5 <- which(INFO_GP$K == 5)
k7 <- which(INFO_GP$K == 7)
k9 <- which(INFO_GP$K == 9)
k11 <- which(INFO_GP$K == 11)
k13 <- which(INFO_GP$K == 13)
k15 <- which(INFO_GP$K == 15)
k17 <- which(INFO_GP$K == 17)

plot(0,0,type = 'l', xlim = c(.7,.85),ylim = c(.7,.85),xlab = "Recall",  ylab = "Precision", main = "Experimento 6 - PCA (26)\nOtimizando Precision, Recall e Accuracy para Spindles")
draw_ellipses_ROC(INFO_GP, k3, 19, 'K = 3')
draw_ellipses_ROC(INFO_GP, k5, 19, 'K = 5')
draw_ellipses_ROC(INFO_GP, k7, 19, 'K = 7')
draw_ellipses_ROC(INFO_GP, k9, 19, 'K = 9')
draw_ellipses_ROC(INFO_GP, k11, 19, 'K = 11')
draw_ellipses_ROC(INFO_GP, k13, 19, 'K = 13')
draw_ellipses_ROC(INFO_GP, k15, 19, 'K = 15')
draw_ellipses_ROC(INFO_GP, k17, 19, 'K = 17')

#legend('bottomright',fill=c("green","red"),legend=c("Spindle","Non-Spindle"), inset=.02,bty = "n")

draw_ellipses_ACC(INFO_GP, k3, 19, 'K = 3')
draw_ellipses_ACC(INFO_GP, k5, 19, 'K = 5')
draw_ellipses_ACC(INFO_GP, k7, 19, 'K = 7')
draw_ellipses_ACC(INFO_GP, k9, 19, 'K = 9')
draw_ellipses_ACC(INFO_GP, k11, 19, 'K = 9')
draw_ellipses_ACC(INFO_GP, k13, 19, 'K = 9')
draw_ellipses_ACC(INFO_GP, k15, 19, 'K = 9')
legend('bottomright',fill=c("green","red", "black"),legend=c("Spindle","Non-Spindle", "Model Accuracy"), inset=.02,bty = "n")

par(mfrow = c(1,2))
boxplot(INFO_GP$PPV_S~INFO_GP$K, ylim = c(.73,.84), col = rgb(0,1,0,0.35), border = 'darkgreen', main = "Experimento 8 (25 features, Deep = 35) \n Precision", ylab = 'Precision', xlab = 'K')
boxplot(INFO_GP$PPV_NS~INFO_GP$K, ylim = c(.73,.84), col = rgb(1,0,0,0.35), border = 'red', add = TRUE)
legend('bottomleft',fill=c("green","red"),legend=c("Spindle","Non-Spindle"), inset=.02,bty = "n")

boxplot(INFO_GP$TPR_S~INFO_GP$K, ylim = c(.73,.84), col = rgb(0,1,0,0.35), border = 'darkgreen', main = "Experimento 8 (25 features, Deep = 35) \n Recall", ylab = 'Recall', xlab = 'K')
boxplot(INFO_GP$TPR_NS~INFO_GP$K, ylim = c(.73,.84), col = rgb(1,0,0,0.35), border = 'red', add = TRUE)
legend('bottomleft',fill=c("green","red"),legend=c("Spindle","Non-Spindle"), inset=.02,bty = "n")

