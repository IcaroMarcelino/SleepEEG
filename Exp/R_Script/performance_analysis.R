plot_classifiers <- function(eng, type){
  library(readr)
  if(type == 1){
    subtitle = 'Janelas disjuntas de 2 s'
    DT  <- read_csv("~/Desktop/TG/Dados_Sono/SleepEEG/Exp/Analysis/DT.csv")
    KNN <- read_csv("~/Desktop/TG/Dados_Sono/SleepEEG/Exp/Analysis/KNN_1.csv")
    MLP <- read_csv("~/Desktop/TG/Dados_Sono/SleepEEG/Exp/Analysis/MLP_13_2.csv")
    NB  <- read_csv("~/Desktop/TG/Dados_Sono/SleepEEG/Exp/Analysis/NB.csv")
    SVM <- read_csv("~/Desktop/TG/Dados_Sono/SleepEEG/Exp/Analysis/SVM.csv")
  }else{
    subtitle = 'Identificação usando somente os classificadores'
    DT  <- read_csv("~/Desktop/TG/Dados_Sono/SleepEEG/Exp/Analysis2/DT.csv")
    KNN <- read_csv("~/Desktop/TG/Dados_Sono/SleepEEG/Exp/Analysis2/KNN_1.csv")
    MLP <- read_csv("~/Desktop/TG/Dados_Sono/SleepEEG/Exp/Analysis2/MLP_13_2.csv")
    NB  <- read_csv("~/Desktop/TG/Dados_Sono/SleepEEG/Exp/Analysis2/NB.csv")
    SVM <- read_csv("~/Desktop/TG/Dados_Sono/SleepEEG/Exp/Analysis2/SVM.csv")
  }
  MLP1 <- MLP[which(MLP$Activattion == 'relu' & MLP$`#Neurons` == 15),]
  KNN1 <- KNN[which(KNN$K == 5),]
  SVM1 <- SVM[which(SVM$Kernel == 'linear'),]
  
  SVM1_B  <- SVM1[which(SVM1$Balanced == 1),]
  SVM1_NB <- SVM1[which(SVM1$Balanced == 0),]
  MLP1_B <- MLP1[which(MLP1$Balanced == 1),]
  MLP1_NB<- MLP1[which(MLP1$Balanced == 0),]
  KNN1_B <- KNN1[which(KNN1$Balanced == 1),]
  KNN1_NB<- KNN1[which(KNN1$Balanced == 0),]
  DT_B   <- DT[which(DT$Balanced == 1),]
  DT_NB  <- DT[which(DT$Balanced == 0),]
  NB_B  <- NB[which(NB$Balanced == 1),]
  NB_NB <- NB[which(NB$Balanced == 0),]
  
  if (!eng){
    boxplot(SVM1_B$AUC0, NB_B$AUC0, KNN1_B$AUC0, DT_B$AUC0, MLP1_B$AUC0, ylim = c(0.45,1), main = paste("Área abaixo da curva ROC\nSomente classificadores", subtitle, sep = ' - ') , ylab = "AUC", xlab = "Classificador", col = rgb(0,1,0,.5), border = 'darkgreen')
    boxplot(SVM1_NB$AUC0, NB_NB$AUC0, KNN1_NB$AUC0, DT_NB$AUC0, MLP1_NB$AUC0, ylim = c(0.45,1), main = paste("Área abaixo da curva ROC\nSomente classificadores", subtitle, sep = ' - '), ylab = "AUC", xlab = "Classificador", col = rgb(1,0,0,.5), border = 'red', add = T)
    axis(1, at=1:5, labels=c('SVM', 'NB', 'KNN','DT', 'MLP'))
    legend('topleft', c('Classes balanceadas', 'Classes desbalanceadas'), fill = c('green', 'red'), inset = .0, bty = 'n')
  }else{
    boxplot(SVM1_B$AUC0, NB_B$AUC0, KNN1_B$AUC0, DT_B$AUC0, MLP1_B$AUC0, ylim = c(0.45,1), main = paste("Area Under ROC Curve\nOnly Classifiers", subtitle, sep = ' - '), ylab = "AUC", xlab = "Classifier", col = rgb(0,1,0,.5), border = 'darkgreen')
    boxplot(SVM1_NB$AUC0, NB_NB$AUC0, KNN1_NB$AUC0, DT_NB$AUC0, MLP1_NB$AUC0, ylim = c(0.45,1), main = paste("Area Under ROC Curve\nOnly Classifiers", subtitle, sep = ' - '), ylab = "AUC", xlab = "Classifier", col = rgb(1,0,0,.5), border = 'red', add = T)
    axis(1, at=1:5, labels=c('SVM', 'NB', 'KNN','DT', 'MLP'))
    legend('topleft', c('Balanced Classes', 'Unbalanced Classes'), fill = c('green', 'red'), inset = .0, bty = 'n')
  }
}

plot_KNN <- function(eng, type){
  library(readr)
  if(!eng){
    if(type == 1){
      KNN <- read_csv("~/Desktop/TG/Dados_Sono/SleepEEG/Exp/Analysis2/KNN_1.csv")
      subtitle = 'Janelas disjuntas de 2 s'
    }else{
      KNN <- read_csv("~/Desktop/TG/Dados_Sono/SleepEEG/Exp/Analysis2/KNN_1.csv")
      subtitle = ''
    }
    KNN_B <- KNN[which(KNN$Balanced == 1),]
    KNN_NB<- KNN[which(KNN$Balanced == 0),]
    boxplot(KNN_B$AUC0~KNN_B$K, ylim = c(.5,.7),main = paste("Área abaixo da curva ROC\nClassificação com KNN", subtitle), ylab = "AUC", xlab = "K", col = rgb(0,1,0,.5), border = 'darkgreen')
    boxplot(KNN_NB$AUC0~KNN_NB$K, main = paste("Área abaixo da curva ROC\nClassificação com KNN", subtitle), xlab = "K", col = rgb(1,0,0,.5), border = 'red', add = T)
    legend('topleft', c('Classes balanceadas', 'Classes desbalanceadas'), fill = c('green', 'red'), inset = .0, bty = 'n')
  }else{
    KNN_B <- KNN[which(KNN$Balanced == 1),]
    KNN_NB<- KNN[which(KNN$Balanced == 0),]
    boxplot(KNN_B$AUC0~KNN_B$K, ylim = c(.5,.7),main = "Area Under ROC Curve\nKNN", ylab = "AUC", xlab = "K", col = rgb(0,1,0,.5), border = 'darkgreen')
    boxplot(KNN_NB$AUC0~KNN_NB$K, main = "Area Under ROC Curve\nKNN", ylab = "AUC", xlab = "K", col = rgb(1,0,0,.5), border = 'red', add = T)
    legend('topleft', c('Balanced Classes', 'Unbalanced Classes'), fill = c('green', 'red'), inset = .0, bty = 'n')
  }
}

plot_SVM <- function(eng, type){
  library(readr)
  
  if(type == 1){
    SVM <- read_csv("~/Desktop/TG/Dados_Sono/SleepEEG/Exp/Analysis/SVM.csv")
    subtitle = 'Janelas disjuntas de 2 s'
  }else{
    SVM <- read_csv("~/Desktop/TG/Dados_Sono/SleepEEG/Exp/Analysis2/SVM.csv")
    subtitle = ''
  }
  if(!eng){
    SVM_B <- SVM[which(SVM$Balanced == 1),]
    SVM_NB<- SVM[which(SVM$Balanced == 0),]
    boxplot(SVM_B$AUC0~SVM_B$Kernel, ylim = c(.5,.8),main = paste("Área abaixo da curva ROC\nClassificação com SVM", subtitle), ylab = "AUC", xlab = "Kernel", col = rgb(0,1,0,.5), border = 'darkgreen')
    boxplot(SVM_NB$AUC0~SVM_NB$Kernel, main = paste("Área abaixo da curva ROC\nClassificação com SVM", subtitle), ylab = "AUC", xlab = "Kernel", col = rgb(1,0,0,.5), border = 'red', add = T)
    legend('topleft', c('Classes balanceadas', 'Classes desbalanceadas'), fill = c('green', 'red'), inset = .0, bty = 'n')
  }else{
    SVM_B <- SVM[which(SVM$Balanced == 1),]
    SVM_NB<- SVM[which(SVM$Balanced == 0),]
    boxplot(SVM_B$AUC0~SVM_B$Kernel, ylim = c(.5,.8),main = "Área under ROC curve\nSVM", ylab = "AUC", xlab = "Kernel", col = rgb(0,1,0,.5), border = 'darkgreen')
    boxplot(SVM_NB$AUC0~SVM_NB$Kernel, main = "Área under ROC curve\nClassificação com SVM", ylab = "AUC", xlab = "Kernel", col = rgb(1,0,0,.5), border = 'red', add = T)
    legend('topleft', c('Balanced Classes', 'Unbalanced Classes'), fill = c('green', 'red'), inset = .0, bty = 'n')
  }
}

plot_MLP <- function(eng){
  library(readr)
  MLP <- read_csv("~/Desktop/TG/Dados_Sono/SleepEEG/Exp/Analysis2/MLP_13_2.csv")

  if(!eng){
    par(mfrow = c(1,2))
    MLP_B <- MLP[which(MLP$Balanced == 1 & MLP$Activattion == 'logistic'),]
    MLP_NB<- MLP[which(MLP$Balanced == 0 & MLP$Activattion == 'logistic'),]
    boxplot(MLP_B$AUC0~MLP_B$`#Neurons`, ylim = c(.5,.8), main = "Área abaixo da curva ROC\nMLP - Ativação: Função Logística", ylab = "AUC", xlab = "Número de neurônios", col = rgb(0,1,0,.5), border = 'darkgreen')
    boxplot(MLP_NB$AUC0~MLP_NB$`#Neurons`, col = rgb(1,0,0,.5), border = 'red', add = T)
    legend('topleft', c('Classes balanceadas', 'Classes desbalanceadas'), fill = c('green', 'red'), inset = .0, bty = 'n')
    
    MLP_B <- MLP[which(MLP$Balanced == 1 & MLP$Activattion == 'relu'),]
    MLP_NB<- MLP[which(MLP$Balanced == 0 & MLP$Activattion == 'relu'),]
    boxplot(MLP_B$AUC0~MLP_B$`#Neurons`, ylim = c(.5,.8), main = "Área abaixo da curva ROC\nMLP - Ativação: ReLU", ylab = "AUC", xlab = "Número de neurônios", col = rgb(0,1,0,.5), border = 'darkgreen')
    boxplot(MLP_NB$AUC0~MLP_NB$`#Neurons`, col = rgb(1,0,0,.5), border = 'red', add = T)
  }
  else{
    par(mfrow = c(1,2))
    MLP_B <- MLP[which(MLP$Balanced == 1 & MLP$Activattion == 'logistic'),]
    MLP_NB<- MLP[which(MLP$Balanced == 0 & MLP$Activattion == 'logistic'),]
    boxplot(MLP_B$AUC0~MLP_B$`#Neurons`, ylim = c(.5,.8), main = "Area Under ROC Curve\nMLP - Activation: Logistic", ylab = "AUC", xlab = "Number of Neurons", col = rgb(0,1,0,.5), border = 'darkgreen')
    boxplot(MLP_NB$AUC0~MLP_NB$`#Neurons`, col = rgb(1,0,0,.5), border = 'red', add = T)
    legend('topleft', c('Balancead Classes', 'Unbalanced Classes'), fill = c('green', 'red'), inset = .0, bty = 'n')
    
    MLP_B <- MLP[which(MLP$Balanced == 1 & MLP$Activattion == 'relu'),]
    MLP_NB<- MLP[which(MLP$Balanced == 0 & MLP$Activattion == 'relu'),]
    boxplot(MLP_B$AUC0~MLP_B$`#Neurons`, ylim = c(.5,.8), main = "Area Under ROC Curve\nMLP - Activation: ReLU", ylab = "AUC", xlab = "Number of Neurons", col = rgb(0,1,0,.5), border = 'darkgreen')
    boxplot(MLP_NB$AUC0~MLP_NB$`#Neurons`, col = rgb(1,0,0,.5), border = 'red', add = T)
  }
}

plot_MLP2 <- function(eng){
  library(readr)
  MLP <- read_csv("~/Desktop/TG/Dados_Sono/SleepEEG/Exp/Analysis2/MLP_13_3.csv")
  
  if(!eng){
    par(mfrow = c(1,2))
    MLP_B <- MLP[which(MLP$Balanced == 1 & MLP$Activattion == 'tanh'),]
    MLP_NB<- MLP[which(MLP$Balanced == 0 & MLP$Activattion == 'tanh'),]
    boxplot(MLP_B$AUC0~MLP_B$`#Neurons`, ylim = c(.45,.95), main = "Área abaixo da curva ROC\nMLP - Ativação: Função tanh", ylab = "AUC", xlab = "Número de neurônios", col = rgb(0,1,0,.5), border = 'darkgreen')
    boxplot(MLP_NB$AUC0~MLP_NB$`#Neurons`, col = rgb(1,0,0,.5), border = 'red', add = T)
    legend('topleft', c('Classes balanceadas', 'Classes desbalanceadas'), fill = c('green', 'red'), inset = .0, bty = 'n')
    
    MLP_B <- MLP[which(MLP$Balanced == 1 & MLP$Activattion == 'relu'),]
    MLP_NB<- MLP[which(MLP$Balanced == 0 & MLP$Activattion == 'relu'),]
    boxplot(MLP_B$AUC0~MLP_B$`#Neurons`, ylim = c(.45,.95), main = "Área abaixo da curva ROC\nMLP - Ativação: ReLU", ylab = "AUC", xlab = "Número de neurônios", col = rgb(0,1,0,.5), border = 'darkgreen')
    boxplot(MLP_NB$AUC0~MLP_NB$`#Neurons`, col = rgb(1,0,0,.5), border = 'red', add = T)
  }
  else{
    par(mfrow = c(1,2))
    MLP_B <- MLP[which(MLP$Balanced == 1 & MLP$Activattion == 'logistic'),]
    MLP_NB<- MLP[which(MLP$Balanced == 0 & MLP$Activattion == 'logistic'),]
    boxplot(MLP_B$AUC0~MLP_B$`#Neurons`, ylim = c(.45,.8), main = "Area Under ROC Curve\nMLP - Activation: Logistic", ylab = "AUC", xlab = "Number of Neurons", col = rgb(0,1,0,.5), border = 'darkgreen')
    boxplot(MLP_NB$AUC0~MLP_NB$`#Neurons`, col = rgb(1,0,0,.5), border = 'red', add = T)
    legend('topleft', c('Balancead Classes', 'Unbalanced Classes'), fill = c('green', 'red'), inset = .0, bty = 'n')
    
    MLP_B <- MLP[which(MLP$Balanced == 1 & MLP$Activattion == 'relu'),]
    MLP_NB<- MLP[which(MLP$Balanced == 0 & MLP$Activattion == 'relu'),]
    boxplot(MLP_B$AUC0~MLP_B$`#Neurons`, ylim = c(.45,.8), main = "Area Under ROC Curve\nMLP - Activation: ReLU", ylab = "AUC", xlab = "Number of Neurons", col = rgb(0,1,0,.5), border = 'darkgreen')
    boxplot(MLP_NB$AUC0~MLP_NB$`#Neurons`, col = rgb(1,0,0,.5), border = 'red', add = T)
  }
}

plot_performance <- function(deep_max, eng, subt){
  library(readr)
  BL  <- read_csv(paste("~/Desktop/TG/Dados_Sono/SleepEEG/Exp/T", deep_max,"b/infoGP.csv", sep = ''))
  #NBL <- read_csv(paste("~/Desktop/TG/Dados_Sono/SleepEEG/Exp/T", deep_max,"/infoGP.csv", sep = ''))
  
  #BL <- BL[which(BL$`DEEP MAX` == deep_max),]
  #NBL <- NBL[which(NBL$`DEEP MAX` == deep_max),]
  
  # Test plot
  # boxplot(BL$AUC~BL$classifier, ylim = c(.5,.85),main = "Área abaixo da curva ROC\nClassificação de atributos GP", ylab = "AUC", xlab = "Classificador", col = rgb(0,1,0,.5), border = 'darkgreen')
  # boxplot(NBL$AUC~NBL$classifier, col = rgb(1,0,0,.5), border = 'red', add = T)
  
  SVM_B <- BL[which(BL$classifier == 'svm'),]
  #SVM_NB<-NBL[which(NBL$classifier== 'svm'),]
  NB_B <- BL[which(BL$classifier == 'nb'),]
  #NB_NB<-NBL[which(NBL$classifier== 'nb'),]
  KNN_B <- BL[which(BL$classifier == 'knn'),]
  #KNN_NB<-NBL[which(NBL$classifier== 'knn'),]
  DT_B <- BL[which(BL$classifier == 'dt'),]
  #DT_NB<-NBL[which(NBL$classifier== 'dt'),]
  MLP_B <- BL[which(BL$classifier == 'mlp'),]
  #MLP_NB<-NBL[which(NBL$classifier== 'mlp'),]
  #KMEANS_B <- BL[which(BL$classifier == 'kmeans'),]
  #KMEANS_NB<-NBL[which(NBL$classifier== 'kmeans'),]
  
  # if (!eng){
  #  boxplot(KNN_B$AUC~KNN_B$P1, ylim = c(.45,.85),main = "Área abaixo da curva ROC\nGP+KNN", ylab = "AUC", xlab = "K", col = rgb(0,1,0,.5), border = 'darkgreen')
  #  boxplot(KNN_NB$AUC~KNN_NB$P1, main = "Área abaixo da curva ROC\nGP+KNN", ylab = "AUC", xlab = "K", col = rgb(1,0,0,.5), border = 'red', add = T)
  # }else{
  #  boxplot(KNN_B$AUC~KNN_B$P1, ylim = c(.45,.85),main = "Área Under ROC Curve\nGP+KNN", ylab = "AUC", xlab = "K", col = rgb(0,1,0,.5), border = 'darkgreen')
  #  boxplot(KNN_NB$AUC~KNN_NB$P1, main = "Área Under ROC Curve\nGP+KNN", ylab = "AUC", xlab = "K", col = rgb(1,0,0,.5), border = 'red', add = T)
  # }
  
  KNN1_B <- KNN_B[which(KNN_B$P1 == 5),]
  #KNN1_NB <- KNN_NB[which(KNN_NB$P1 == 5),]
  
  if (!eng){
    boxplot(SVM_B$AUC, NB_B$AUC, KNN1_B$AUC, DT_B$AUC, MLP_B$AUC, ylim = c(0.45,1), main = paste("Área abaixo da curva ROC - Construção de Atributos com GP\n",subt, sep = ''), ylab = "AUC", xlab = "Classificador", col = rgb(0,1,0,.5), border = 'darkgreen')
    #boxplot(SVM_NB$AUC, NB_NB$AUC, KNN1_NB$AUC, DT_NB$AUC, MLP_NB$AUC, KMEANS_NB$AUC, ylim = c(0.45,.95), col = rgb(1,0,0,.5), border = 'red', add = T)
    axis(1, at=1:5, labels=c('SVM', 'NB', 'KNN','DT', 'MLP'))
    #legend('bottom', c('Classes balanceadas', 'Classes desbalanceadas'), fill = c('green', 'red'), inset = .0, bty = 'n')
  }else{
    boxplot(SVM_B$AUC, NB_B$AUC, KNN1_B$AUC, DT_B$AUC, MLP_B$AUC, ylim = c(0.45,1), main =  paste("Area Under ROC Curve\nGP Feature Construction",subt, sep = ''), ylab = "AUC", xlab = "Classifier", col = rgb(0,1,0,.5), border = 'darkgreen')
    boxplot(SVM_NB$AUC, NB_NB$AUC, KNN1_NB$AUC, DT_NB$AUC, MLP_NB$AUC, ylim = c(0.45,1), col = rgb(1,0,0,.5), border = 'red', add = T)
    axis(1, at=1:5, labels=c('SVM', 'NB', 'KNN','DT', 'MLP', 'K-means'))
    legend('topleft', c('Balanced Classes', 'Unbalanced Classes'), fill = c('green', 'red'), inset = .0, bty = 'n')
  }
}

plot_performance_all <- function(deep_max, eng, subt){
  library(readr)
  BL  <- read_csv(paste("~/Desktop/TG/Dados_Sono/SleepEEG/Exp/T", deep_max,"b/infoGP.csv", sep = ''))
  NBL <- read_csv(paste("~/Desktop/TG/Dados_Sono/SleepEEG/Exp/T", deep_max,"/infoGP.csv", sep = ''))
  
  #BL <- BL[which(BL$`DEEP MAX` == deep_max),]
  #NBL <- NBL[which(NBL$`DEEP MAX` == deep_max),]
  
  # Test plot
  # boxplot(BL$AUC~BL$classifier, ylim = c(.5,.85),main = "Área abaixo da curva ROC\nClassificação de atributos GP", ylab = "AUC", xlab = "Classificador", col = rgb(0,1,0,.5), border = 'darkgreen')
  # boxplot(NBL$AUC~NBL$classifier, col = rgb(1,0,0,.5), border = 'red', add = T)
  
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
  #KMEANS_B <- BL[which(BL$classifier == 'kmeans'),]
  #KMEANS_NB<-NBL[which(NBL$classifier== 'kmeans'),]
  
  #if (!eng){
  #  boxplot(KNN_B$AUC~KNN_B$P1, ylim = c(.45,.85),main = "Área abaixo da curva ROC\nGP+KNN", ylab = "AUC", xlab = "K", col = rgb(0,1,0,.5), border = 'darkgreen')
  #  boxplot(KNN_NB$AUC~KNN_NB$P1, main = "Área abaixo da curva ROC\nGP+KNN", ylab = "AUC", xlab = "K", col = rgb(1,0,0,.5), border = 'red', add = T)
  #}else{
  #  boxplot(KNN_B$AUC~KNN_B$P1, ylim = c(.45,.85),main = "Área Under ROC Curve\nGP+KNN", ylab = "AUC", xlab = "K", col = rgb(0,1,0,.5), border = 'darkgreen')
  #  boxplot(KNN_NB$AUC~KNN_NB$P1, main = "Área Under ROC Curve\nGP+KNN", ylab = "AUC", xlab = "K", col = rgb(1,0,0,.5), border = 'red', add = T)
  # }
  
  KNN1_B <- KNN_B[which(KNN_B$P1 == 5),]
  KNN1_NB <- KNN_NB[which(KNN_NB$P1 == 5),]
  
  if (!eng){
    boxplot(SVM_B$AUC, NB_B$AUC, KNN1_B$AUC, DT_B$AUC, MLP_B$AUC, ylim = c(0.45,1), main = paste("Área abaixo da curva ROC - Construção de Atributos com GP\n",subt, sep = ''), ylab = "AUC", xlab = "Classificador", col = rgb(0,1,0,.5), border = 'darkgreen')
    boxplot(SVM_NB$AUC, NB_NB$AUC, KNN1_NB$AUC, DT_NB$AUC, MLP_NB$AUC, ylim = c(0.45,.1), col = rgb(1,0,0,.5), border = 'red', add = T)
    axis(1, at=1:5, labels=c('SVM', 'NB', 'KNN','DT', 'MLP'))
    legend('bottom', c('Classes balanceadas', 'Classes desbalanceadas'), fill = c('green', 'red'), inset = .0, bty = 'n')
  }else{
    boxplot(SVM_B$AUC, NB_B$AUC, KNN1_B$AUC, DT_B$AUC, MLP_B$AUC, ylim = c(0.45,1), main =  paste("Area Under ROC Curve\nGP Feature Construction",subt, sep = ''), ylab = "AUC", xlab = "Classifier", col = rgb(0,1,0,.5), border = 'darkgreen')
    boxplot(SVM_NB$AUC, NB_NB$AUC, KNN1_NB$AUC, DT_NB$AUC, MLP_NB$AUC, ylim = c(0.45,1), col = rgb(1,0,0,.5), border = 'red', add = T)
    axis(1, at=1:5, labels=c('SVM', 'NB', 'KNN','DT', 'MLP'))
    legend('topleft', c('Balanced Classes', 'Unbalanced Classes'), fill = c('green', 'red'), inset = .0, bty = 'n')
  }
}



precision <- function(tp, fp){
  return(tp/(tp+fp))
}

recall <- function(tp, fn){
  return(tp/(tp+fn))
}

f1_score <- function(tp, fp, fn){
  return(2*tp/(2*tp+fn+fp))
}

