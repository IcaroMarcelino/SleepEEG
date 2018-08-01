library(readr)
DT  <- read_csv("Downloads/Dados_Sono/GP/SleepEEG/Exp/Analysis/DT.csv")
KNN <- read_csv("Downloads/Dados_Sono/GP/SleepEEG/Exp/Analysis/KNN.csv")
MLP <- read_csv("Downloads/Dados_Sono/GP/SleepEEG/Exp/Analysis/MLP_13_2.csv")
NB  <- read_csv("Downloads/Dados_Sono/GP/SleepEEG/Exp/Analysis/NB.csv")
SVM <- read_csv("Downloads/Dados_Sono/GP/SleepEEG/Exp/Analysis/SVM.csv")

MLP1 <- MLP[which(MLP$Activattion == 'relu' & MLP$`#Neurons` == 15),]
KNN1 <- KNN[which(KNN$K == 3),]

SVM_B  <- SVM[which(SVM$Balanced == 1),]
SVM_NB <- SVM[which(SVM$Balanced == 0),]
MLP1_B <- MLP1[which(MLP1$Balanced == 1),]
MLP1_NB<- MLP1[which(MLP1$Balanced == 0),]
KNN1_B <- KNN1[which(KNN1$Balanced == 1),]
KNN1_NB<- KNN1[which(KNN1$Balanced == 0),]
DT_B   <- DT[which(DT$Balanced == 1),]
DT_NB  <- DT[which(DT$Balanced == 0),]
NB_B  <- NB[which(NB$Balanced == 1),]
NB_NB <- NB[which(NB$Balanced == 0),]

boxplot(SVM_B$AUC0, NB_B$AUC0, KNN1_B$AUC0, DT_B$AUC0, MLP1_B$AUC0, ylim = c(0.45,0.8), main = "Área abaixo da curva ROC", ylab = "AUC", xlab = "Classificador", col = rgb(0,1,0,.5), border = 'darkgreen')
boxplot(SVM_NB$AUC0, NB_NB$AUC0, KNN1_NB$AUC0, DT_NB$AUC0, MLP1_NB$AUC0, ylim = c(0.45,0.8), main = "Área abaixo da curva ROC", ylab = "AUC", xlab = "Classificador", col = rgb(1,0,0,.5), border = 'red', add = T)
axis(1, at=1:5, labels=c('SVM', 'NB', 'KNN','DT', 'MLP'))
legend('topleft', c('Classes balanceadas', 'Classes desbalanceadas'), fill = c('green', 'red'), inset = .0, bty = 'n')

KNN_B <- KNN[which(KNN$Balanced == 1),]
KNN_NB<- KNN[which(KNN$Balanced == 0),]
boxplot(KNN_B$AUC0~KNN_B$K, ylim = c(.55,.7),main = "Área abaixo da curva ROC\nKNN", ylab = "AUC", xlab = "K", col = rgb(0,1,0,.5), border = 'darkgreen')
boxplot(KNN_NB$AUC0~KNN_NB$K, main = "Área abaixo da curva ROC\nKNN", ylab = "AUC", xlab = "K", col = rgb(1,0,0,.5), border = 'red', add = T)

par(mfrow = c(1,2))
MLP_B <- MLP[which(MLP$Balanced == 1 & MLP$Activattion == 'logistic'),]
MLP_NB<- MLP[which(MLP$Balanced == 0 & MLP$Activattion == 'logistic'),]
boxplot(MLP_B$AUC0~MLP_B$`#Neurons`, main = "Área abaixo da curva ROC\nMLP - Ativação: Função Logística", ylab = "AUC", xlab = "Número de neurônios", col = rgb(0,1,0,.5), border = 'darkgreen')
boxplot(MLP_NB$AUC0~MLP_NB$`#Neurons`, col = rgb(1,0,0,.5), border = 'red', add = T)

MLP_B <- MLP[which(MLP$Balanced == 1 & MLP$Activattion == 'relu'),]
MLP_NB<- MLP[which(MLP$Balanced == 0 & MLP$Activattion == 'relu'),]
boxplot(MLP_B$AUC0~MLP_B$`#Neurons`, main = "Área abaixo da curva ROC\nMLP - Ativação: ReLU", ylab = "AUC", xlab = "Número de neurônios", col = rgb(0,1,0,.5), border = 'darkgreen')
boxplot(MLP_NB$AUC0~MLP_NB$`#Neurons`, col = rgb(1,0,0,.5), border = 'red', add = T)

legend('bottomright', c('Classes balanceadas', 'Classes desbalanceadas'), fill = c('green', 'red'), inset = .0, bty = 'n')
