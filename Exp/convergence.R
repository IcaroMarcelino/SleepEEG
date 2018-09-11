get_pair <- function(temp){
  library(stringr)
  temp1 <- temp
  temp2 <- temp
  for(i in (1:length(temp))){
    temp[i] = str_replace(temp[i], "\\[ ", "")
    temp[i] = str_replace(temp[i], "\\[", "")
    temp[i] = str_replace(temp[i], "\\]", "")
    temp[i] = str_replace(temp[i], " \\]", "")
    x = str_split(temp[i], " ")
    x = x[[1]][which(x[[1]] != "")]
    #x = x[[1]][which(x[[1]] != " ")]
    #print(x)
    y = as.numeric(x)
    #print(y[2])
    temp1[i] = y[1]
    temp2[i] = y[2]
  }
  r <- NULL
  r$A <- as.numeric(temp1)
  r$B <- as.numeric(temp2)
  return(r)
}

evolution <- function(folder,files, iter, cl){
  library(readr)
  aux <- matrix(nrow = iter, ncol = length(files))
  est <- c(rep(0,each=iter))
  desvpos <- c(rep(0,each=iter))
  desvneg <- c(rep(0,each=iter))
  desvmax <- c(rep(0,each=iter))
  desvmin <- c(rep(0,each=iter))
  for (i in 1:length(files)){
    f <- read_delim(paste(folder,files[i],sep = ''),"\t", escape_double = FALSE, trim_ws = TRUE, skip = 2)
    a <- get_pair(f$max)
    for (j in 1:iter){
      est[j] <- est[j] + as.numeric(a[[cl]][j]/length(files))
      aux[j,i] <- as.numeric(a[[cl]][j])
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

get_evolution <- function(deep_max, b, classifier, param, iter, cl){
  folder = paste("T", deep_max, b, "/log/",sep='')
  name = paste("*", classifier, param, "*",sep='')
  files  = list.files(path = folder, pattern = name)
  r <- evolution(folder, files, iter, cl)
  return(r)
}

print_convergence <- function(deep_max, b, classifier, param){
  cl = 1
  iter = 300
  d <- get_evolution(deep_max, b, classifier, param, iter, cl)
  
  balanced = 'Balanceadas)'
  if(b==''){
    balanced = 'Desbalanceadas)'
  }
  
  par(mfrow=c(1,2))
  yl = c(0,0.6)
  metrica = 'F1 score'
  plot(0,0,type = "l", ylim = yl, xlim = c(0,300), main = paste("Treinamento do modelo\nGP + ", str_to_upper(classifier), "\n(Classes ", balanced ,sep = '') , xlab = "Gerações", ylab = metrica)
  polygon(c(1:300,rev(1:300)), c(d$desvpos, rev(d$desvneg)), col = "grey", border = 'grey')
  lines(0:299, d$custos, ylim =yl, xlim = c(0,299), type = "l", col = "red", lwd = 3)
  box()
  lines(1:300, d$desvmax, col = "blue", pch = 2, lty = 6)
  lines(1:300, d$desvmin, col = "darkgreen", pch = 2, lty = 6)
  
  cl = 2
  d <- get_evolution(deep_max, b, classifier, param, iter, cl)
  yl = c(0,120)
  metrica = 'Verdadeiros Positivos'
  plot(0,0,type = "l", ylim = yl, xlim = c(0,300), main = paste("Treinamento do modelo\nGP + ", str_to_upper(classifier), "\n(Classes ", balanced , sep = '') , xlab = "Gerações", ylab = metrica)
  polygon(c(1:300,rev(1:300)), c(d$desvpos, rev(d$desvneg)), col = "grey", border = 'grey')
  lines(0:299, d$custos, ylim =yl, xlim = c(0,299), type = "l", col = "red", lwd = 3)
  box()
  lines(1:300, d$desvmax, col = "blue", pch = 2, lty = 6)
  lines(1:300, d$desvmin, col = "darkgreen", pch = 2, lty = 6)
}

print_args <- function(deep_max, b, classifier, param){
  library(stringr)
  folder = paste("T", deep_max, b, "/best_expr/",sep='')
  name = paste("*", classifier, param, "*",sep='')
  files  = list.files(path = folder, pattern = name)
  
  balanced = 'Balanceadas)'
  if(b==''){
    balanced = 'Desbalanceadas)'
  }
  
  args_counter <- matrix(nrow = 75, ncol = 2)
  args_counter[,2] <- 0
  for(i in 1:75){
    args_counter[i,1] <- paste('ARG', i-1, sep = '')
  }
  for(j in 1:length(files)){
    y <- read.table(paste(folder, files[j], sep = ''), header = FALSE, sep = '*')
    
    for(i in 1:75){
      n = as.numeric(str_count(y[[1]],args_counter[i,1]))
      args_counter[i,2] <-as.numeric(args_counter[i,2]) + n
    }
  }
  barplot(as.numeric(args_counter[,2]))
}

ro
deep_max = 10
b = 'b'
classifier = 'svm'
param = ''

print_convergence(deep_max,b,classifier,param)

deep_max = 10
b = 'b'
classifier = 'mlp'
param = ''


print_args(deep_max, b, classifier, param)
print_dim(deep_max, b, classifier, param)
