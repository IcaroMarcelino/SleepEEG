Ex2 <- read_csv("EXP_2/INFO_GP.csv", col_names = FALSE)
colnames(Ex2) <- c("Tamanho Máximo", "K", "Execução", 
                   "Precision (Spindle)", "Recall (Spíndle)",
                   "Precision (Spindle)1", "Precision (Não Spindle)", 
                   "Recall (Spindle)1", "Recall (Não Spindle)", 
                   "F-score (Spindle)", "F-score (Não Spindle)", 
                   "Support (Spindle)", "Suport (Não Spindle)", 
                   "Altura do Melhor Indivíduo", "Tempo de treinamento")

par(mfrow = c(1,3))
boxplot(Ex2$`Precision (Spindle)`~Ex2$K, col = rgb(0,1,0,.3), border = 'darkgreen', ylim = c(0,1), 
        main = "Experimento 2\n Precisão", ylab = "Precisão", xlab = "K")
boxplot(Ex2$`Precision (Não Spindle)`~Ex2$K, col = rgb(1,0,0,.3), border = 'red', ylim = c(0,1), 
        add = TRUE)

boxplot(Ex2$`Recall (Spíndle)`~Ex2$K, col = rgb(0,1,0,.3), border = 'darkgreen', ylim = c(0,1), 
        main = "Experimento 2\n Recall", ylab = "Recall", xlab = "K")
boxplot(Ex2$`Recall (Não Spindle)`~Ex2$K, col = rgb(1,0,0,.3), border = 'red', ylim = c(0,1), 
        add = TRUE)

boxplot(Ex2$`F-score (Spindle)`~Ex2$K, col = rgb(0,1,0,.3), border = 'darkgreen', ylim = c(0,1), 
        main = "Experimento 2\n F-Score", ylab = "F-score", xlab = "K")
boxplot(Ex2$`F-score (Não Spindle)`~Ex2$K, col = rgb(1,0,0,.3), border = 'red', ylim = c(0,1), 
        add = TRUE)
legend('right',fill=c("green","red"),legend=c("Spindle","Non-Spindle"), inset=.02,bty = "n")

Ex3 <- read_csv("EXP_3/INFO_GP.csv", col_names = FALSE)
colnames(Ex3) <- c("Tamanho Máximo", "K", "Execução", 
                   "Accuracy", 
                   "Precision (Spindle)", "Precision (Não Spindle)", 
                   "Recall (Spindle)", "Recall (Não Spindle)", 
                   "F-score (Spindle)", "F-score (Não Spindle)", 
                   "Support (Spindle)", "Suport (Não Spindle)", 
                   "Altura do Melhor Indivíduo", "Tempo de treinamento")

par(mfrow = c(1,3))
boxplot(Ex3$`Precision (Spindle)`~Ex3$K, col = rgb(0,1,0,.3), border = 'darkgreen', ylim = c(0,1), 
        main = "Experimento 3\n Precisão", ylab = "Precisão", xlab = "K")
boxplot(Ex3$`Precision (Não Spindle)`~Ex3$K, col = rgb(1,0,0,.3), border = 'red', ylim = c(0,1), 
        add = TRUE)

boxplot(Ex3$`Recall (Spindle)`~Ex3$K, col = rgb(0,1,0,.3), border = 'darkgreen', ylim = c(0,1), 
        main = "Experimento 3\n Recall", ylab = "Recall", xlab = "K")
boxplot(Ex3$`Recall (Não Spindle)`~Ex3$K, col = rgb(1,0,0,.3), border = 'red', ylim = c(0,1), 
        add = TRUE)

boxplot(Ex3$`F-score (Spindle)`~Ex3$K, col = rgb(0,1,0,.3), border = 'darkgreen', ylim = c(0,1), 
        main = "Experimento 3\n F-Score", ylab = "F-score", xlab = "K")
boxplot(Ex3$`F-score (Não Spindle)`~Ex3$K, col = rgb(1,0,0,.3), border = 'red', ylim = c(0,1), 
        add = TRUE)
legend('right',fill=c("green","red"),legend=c("Spindle","Non-Spindle"), inset=.02,bty = "n")

boxplot(Ex3$Accuracy~Ex3$K, ylim = c(0,1), 
        main = "Experimento 3\n Accuracy", ylab = "Accuracy", xlab = "K", add = T)
