library(readr)
DT  <- read_csv("~/Downloads/Dados_Sono/GP/SleepEEG/Exp/Analysis/DT.csv")
KNN <- read_csv("~/Downloads/Dados_Sono/GP/SleepEEG/Exp/Analysis/KNN.csv")
MLP <- read_csv("~/Downloads/Dados_Sono/GP/SleepEEG/Exp/Analysis/MLP_13_2.csv")
NB  <- read_csv("~/Downloads/Dados_Sono/GP/SleepEEG/Exp/Analysis/NB.csv")
SVM <- read_csv("~/Downloads/Dados_Sono/GP/SleepEEG/Exp/Analysis/SVM.csv")

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

boxplot(SVM_B$AUC0, NB_B$AUC0, KNN1_B$AUC0, DT_B$AUC0, MLP1_B$AUC0, ylim = c(0.45,0.85), main = "Área abaixo da curva ROC\nSomente classificadores", ylab = "AUC", xlab = "Classificador", col = rgb(0,1,0,.5), border = 'darkgreen')
boxplot(SVM_NB$AUC0, NB_NB$AUC0, KNN1_NB$AUC0, DT_NB$AUC0, MLP1_NB$AUC0, ylim = c(0.45,0.85), main = "Área abaixo da curva ROC\nSomente classificadores", ylab = "AUC", xlab = "Classificador", col = rgb(1,0,0,.5), border = 'red', add = T)
axis(1, at=1:5, labels=c('SVM', 'NB', 'KNN','DT', 'MLP'))
legend('topleft', c('Classes balanceadas', 'Classes desbalanceadas'), fill = c('green', 'red'), inset = .0, bty = 'n')

boxplot(SVM_B$AUC0, NB_B$AUC0, KNN1_B$AUC0, DT_B$AUC0, MLP1_B$AUC0, ylim = c(0.45,0.85), main = "Area Under ROC Curve\nOnly Classifiers", ylab = "AUC", xlab = "Classifier", col = rgb(0,1,0,.5), border = 'darkgreen')
boxplot(SVM_NB$AUC0, NB_NB$AUC0, KNN1_NB$AUC0, DT_NB$AUC0, MLP1_NB$AUC0, ylim = c(0.45,0.85), main = "Area Under ROC Curve\nOnly Classifiers", ylab = "AUC", xlab = "Classifier", col = rgb(1,0,0,.5), border = 'red', add = T)
axis(1, at=1:5, labels=c('SVM', 'NB', 'KNN','DT', 'MLP'))
legend('topleft', c('Balanced Classes', 'Unbalanced Classes'), fill = c('green', 'red'), inset = .0, bty = 'n')

KNN_B <- KNN[which(KNN$Balanced == 1),]
KNN_NB<- KNN[which(KNN$Balanced == 0),]
boxplot(KNN_B$AUC0~KNN_B$K, ylim = c(.55,.7),main = "Área abaixo da curva ROC\nKNN", ylab = "AUC", xlab = "K", col = rgb(0,1,0,.5), border = 'darkgreen')
boxplot(KNN_NB$AUC0~KNN_NB$K, main = "Área abaixo da curva ROC\nKNN", ylab = "AUC", xlab = "K", col = rgb(1,0,0,.5), border = 'red', add = T)

KNN_B <- KNN[which(KNN$Balanced == 1),]
KNN_NB<- KNN[which(KNN$Balanced == 0),]
boxplot(KNN_B$AUC0~KNN_B$K, ylim = c(.55,.7),main = "Area Under ROC Curve\nKNN", ylab = "AUC", xlab = "K", col = rgb(0,1,0,.5), border = 'darkgreen')
boxplot(KNN_NB$AUC0~KNN_NB$K, main = "Area Under ROC Curve\nKNN", ylab = "AUC", xlab = "K", col = rgb(1,0,0,.5), border = 'red', add = T)

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

par(mfrow = c(1,2))
MLP_B <- MLP[which(MLP$Balanced == 1 & MLP$Activattion == 'logistic'),]
MLP_NB<- MLP[which(MLP$Balanced == 0 & MLP$Activattion == 'logistic'),]
boxplot(MLP_B$AUC0~MLP_B$`#Neurons`, main = "Area Under ROC Curve\nMLP - Activation: Logistic", ylab = "AUC", xlab = "Number of Neurons", col = rgb(0,1,0,.5), border = 'darkgreen')
boxplot(MLP_NB$AUC0~MLP_NB$`#Neurons`, col = rgb(1,0,0,.5), border = 'red', add = T)

MLP_B <- MLP[which(MLP$Balanced == 1 & MLP$Activattion == 'relu'),]
MLP_NB<- MLP[which(MLP$Balanced == 0 & MLP$Activattion == 'relu'),]
boxplot(MLP_B$AUC0~MLP_B$`#Neurons`, main = "Area Under ROC Curve\nMLP - Activation: ReLU", ylab = "AUC", xlab = "Number of Neurons", col = rgb(0,1,0,.5), border = 'darkgreen')
boxplot(MLP_NB$AUC0~MLP_NB$`#Neurons`, col = rgb(1,0,0,.5), border = 'red', add = T)

legend('bottomright', c('Balancead Classes', 'Unbalanced Classes'), fill = c('green', 'red'), inset = .0, bty = 'n')



library(readr)
BL  <- read_csv("~/Downloads/Dados_Sono/GP/SleepEEG/Exp/T2b/infoGP.csv")
NBL <- read_csv("~/Downloads/Dados_Sono/GP/SleepEEG/Exp/T2/infoGP.csv")

boxplot(BL$AUC~BL$classifier, ylim = c(.7,.85),main = "Área abaixo da curva ROC\nClassificação de atributos GP", ylab = "AUC", xlab = "Classificador", col = rgb(0,1,0,.5), border = 'darkgreen')
boxplot(NBL$AUC~NBL$classifier, col = rgb(1,0,0,.5), border = 'red', add = T)

SVM_B <- BL[which(BL$classifier == 'svm'),]
SVM_NB<-NBL[which(NBL$classifier== 'svm'),]
NB_B <- BL[which(BL$classifier == 'nb'),]
NB_NB<-NBL[which(NBL$classifier== 'nb'),]
KNN_B <- BL[which(BL$classifier == 'knn'),]
KNN_NB<-NBL[which(NBL$classifier== 'knn'),]
DT_B <- BL[which(BL$classifier == 'dt'),]
DT_NB<-NBL[which(NBL$classifier== 'dt'),]
MLP_B <- BL[which(BL$classifier == 'mlp'),]
MLP_NB<-NBL[which(NBL$classifier== 'mlp'),]

boxplot(KNN_B$AUC~KNN_B$P1, ylim = c(.45,.85),main = "Área abaixo da curva ROC\nGP+KNN", ylab = "AUC", xlab = "K", col = rgb(0,1,0,.5), border = 'darkgreen')
boxplot(KNN_NB$AUC~KNN_NB$P1, main = "Área abaixo da curva ROC\nGP+KNN", ylab = "AUC", xlab = "K", col = rgb(1,0,0,.5), border = 'red', add = T)

boxplot(KNN_B$AUC~KNN_B$P1, ylim = c(.45,.85),main = "Área Under ROC Curve\nGP+KNN", ylab = "AUC", xlab = "K", col = rgb(0,1,0,.5), border = 'darkgreen')
boxplot(KNN_NB$AUC~KNN_NB$P1, main = "Área Under ROC Curve\nGP+KNN", ylab = "AUC", xlab = "K", col = rgb(1,0,0,.5), border = 'red', add = T)

KNN1_B <- KNN_B[which(KNN_B$P1 == 5),]
KNN1_NB <- KNN_NB[which(KNN_NB$P1 == 5),]

boxplot(SVM_B$AUC, NB_B$AUC, KNN1_B$AUC, DT_B$AUC, MLP_B$AUC, ylim = c(0.45,0.85), main = "Área abaixo da curva ROC\nConstrução de Atributos com GP", ylab = "AUC", xlab = "Classificador", col = rgb(0,1,0,.5), border = 'darkgreen')
boxplot(SVM_NB$AUC, NB_NB$AUC, KNN1_NB$AUC, DT_NB$AUC, MLP_NB$AUC, ylim = c(0.45,0.85), main = "Área abaixo da curva ROC\nConstrução de Atributos com GP", ylab = "AUC", xlab = "Classificador", col = rgb(1,0,0,.5), border = 'red', add = T)
axis(1, at=1:5, labels=c('SVM', 'NB', 'KNN','DT', 'MLP'))
legend('bottomright', c('Classes balanceadas', 'Classes desbalanceadas'), fill = c('green', 'red'), inset = .0, bty = 'n')

boxplot(SVM_B$AUC, NB_B$AUC, KNN1_B$AUC, DT_B$AUC, MLP_B$AUC, ylim = c(0.45,0.85), main = "Area Under ROC Curve\nGP Feature Construction", ylab = "AUC", xlab = "Classifier", col = rgb(0,1,0,.5), border = 'darkgreen')
boxplot(SVM_NB$AUC, NB_NB$AUC, KNN1_NB$AUC, DT_NB$AUC, MLP_NB$AUC, ylim = c(0.45,0.85), main = "Area Under ROC Curve\nGP Feature Construction", ylab = "AUC", xlab = "Classifier", col = rgb(1,0,0,.5), border = 'red', add = T)
axis(1, at=1:5, labels=c('SVM', 'NB', 'KNN','DT', 'MLP'))
legend('bottomright', c('Balanced Classes', 'Unbalanced Classes'), fill = c('green', 'red'), inset = .0, bty = 'n')


KNN_B <- BL[which(BL$classifier == 'knn'),]
KNN_NB<- NBL[which(NBL$classifier == 'knn'),]
boxplot(KNN_B$AUC~KNN_B$P1, ylim = c(.7,.85),main = "Área abaixo da curva ROC\nKNN", ylab = "AUC", xlab = "K", col = rgb(0,1,0,.5), border = 'darkgreen', xlim = c(0.5,4.5))
boxplot(KNN_NB$AUC~KNN_NB$P1, main = "Área abaixo da curva ROC\nKNN", ylab = "AUC", xlab = "K", col = rgb(1,0,0,.5), border = 'red', add = T)
