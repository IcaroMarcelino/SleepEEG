##################################################################################
# generate_eeg_data(excerpt, op, freq, freq_out)
#
# Lê os dados dos 3 canais EEG de um paciente para a base DREAMS Sleep Spindles
# Arquivos necessários no mesmo diretório do fonte:
#   1. Automatic_detection_excerptX.txt -> X = {1,2,3,4,5,6,7,8}
#   2. Visual_scoring1_excerptX.txt -> X = {1,2,3,4,5,6,7,8}
#   3. Visual_scoring2_excerptX.txt -> X = {1,2,3,4,5,6}
#   4. excerptX -> X = {1,2,3,4,5,6,7,8}
# 
# Parâmetros:
#   1. excerpt (int)    - Número do paciente (1 a 8)
#   2. op (bool)        - 1 se o canal central for C3_A1, 0 se o canal for CZ_A1
#   3. freq (float)     - Frequência de amostragem do sinal em Hz
#   4. freq_out (float) - Frequência de amostragem desajada em Hz (oversampling)
#
# Retorno:
# Matriz com 5 colunas representando o sinal: 
#   d[,1] (float) = t
#   d[,2] (float) = C3_A1 ou CZ_A1
#   d[,3] (float) = 01_A1
#   d[,4] (float) = FP1_A1
#   d[,5] (bool)  = 1 para Spindle, 0 para não Spindle
#
# Exemplos:
#   freq_out = 256
#   d1 <- generate_eeg_data(1,1,100,freq_out)
#   d2 <- generate_eeg_data(2,0,200,freq_out)
#   d3 <- generate_eeg_data(3,1,50,freq_out)
#   d4 <- generate_eeg_data(4,0,200,freq_out)
#   d5 <- generate_eeg_data(5,0,200,freq_out)
#   d6 <- generate_eeg_data(6,0,200,freq_out)
#   d7 <- generate_eeg_data(7,0,200,freq_out)
#   d8 <- generate_eeg_data(8,0,200,freq_out)
#
##################################################################################
generate_eeg_data <- function(excerpt, op, freq, freq_out){
  require(edf)
  require(readr)
  require(ppls)
  
  freq_factor = freq_out/freq
  
  #auto <- read_table2(paste("Automatic_detection_excerpt", excerpt, ".txt", sep = ""), col_types = cols(col_double(), col_double()))
  exp1 <- read_table2(paste("Visual_scoring1_excerpt", excerpt, ".txt", sep = ""), col_types = cols(col_double(), col_double()))
  
  if(excerpt != 7 && excerpt != 8 ){
    exp2 <- read_table2(paste("Visual_scoring2_excerpt", excerpt, ".txt", sep = ""), col_types = cols(col_double(), col_double()))
  }
  excerpt_data <- read.edf(paste("excerpt", excerpt, ".edf", sep = ""))
  
  if(op == 1){
    len = length(excerpt_data$signal$C3_A1$t)
    n_samples = len*freq_out/freq
    sp1 <- spline(excerpt_data$signal$C3_A1$t, excerpt_data$signal$C3_A1$data, n=freq_factor*len, method = "fmm")
  }else{
    len = length(excerpt_data$signal$CZ_A1$t)
    n_samples = len*freq_out/freq
    sp1 <- spline(excerpt_data$signal$CZ_A1$t, excerpt_data$signal$CZ_A1$data, n=freq_factor*len, method = "fmm")
  }
  d = matrix(nrow = n_samples, ncol = 5)

  if(freq_factor > 1){
    d[,1] = sp1$x
    d[,2] = sp1$y
    d[,3] = spline(excerpt_data$signal$O1_A1$t, excerpt_data$signal$O1_A1$data, n=freq_factor*len, method = "fmm")$y
    d[,4] = spline(excerpt_data$signal$FP1_A1$t, excerpt_data$signal$FP1_A1$data, n=freq_factor*len, method = "fmm")$y
    d[,5] = 0
  }else{
    if(op == 1){
      d[,1] = excerpt_data$signal$C3_A1$t
      d[,2] = excerpt_data$signal$C3_A1$data
    }else{
      d[,1] = excerpt_data$signal$CZ_A1$t
      d[,2] = excerpt_data$signal$CZ_A1$data
    }
    d[,3] = excerpt_data$signal$O1_A1$data
    d[,4] = excerpt_data$signal$FP1_A1$data
    d[,5] = 0
  }
  #for(i in 1:length(auto[,1,1])){
  for(i in 1:length(exp1[,1,1])){
    if(op == 1){
      #a = which(d[,1] >= as.double(auto[i,1]))
      #b = which(d[,1] <= as.double(auto[i,1]+as.double(auto[i,2])))
      #c1 = intersect(a,b)
      
      a = which(d[,1] >= as.double(exp1[i,1]))
      b = which(d[,1] <= as.double(exp1[i,1]+as.double(exp1[i,2])))
      c2 = intersect(a,b)
      
      #c = union(c1, c2)
      c = c2
      if(excerpt != 7 && excerpt != 8 ){
        a = which(d[,1] >= as.double(exp2[i,1]))
        b = which(d[,1] <= as.double(exp2[i,1]+as.double(exp2[i,2])))
        c3 = intersect(a,b)
        
        c3 = intersect(a,b)
        c = union(c, c3)
        
      }
      
      d[c, 5] = 1
    }else{
      #a = which(d[,1] >= as.double(auto[i,1]))
      #b = which(d[,1] <= as.double(auto[i,1]+as.double(auto[i,2])))
      #c1 = intersect(a,b)
      
      a = which(d[,1] >= as.double(exp1[i,1]))
      b = which(d[,1] <= as.double(exp1[i,1]+as.double(exp1[i,2])))
      c2 = intersect(a,b)
      
      #c = union(c1, c2)
      c = c2
      if(excerpt != 7 && excerpt != 8 ){
        a = which(d[,1] >= as.double(exp2[i,1]))
        b = which(d[,1] <= as.double(exp2[i,1]+as.double(exp2[i,2])))
        
        c3 = intersect(a,b)
        c = union(c, c3)
      }
      d[c, 5] = 1
    }
  }
  
  #print(sum(d[,5])*100/length(d[,5]))
  #write.table(d, file = paste("ex", excerpt, ".csv", sep = ""), quote = FALSE, sep = ',', row.names = FALSE, col.names = FALSE)
  
  return(d)
}

##################################################################################
# separate_sleep_eeg_band(eeg_data)
# 
# Aplica filtro passa banda 3 canais EEG de um paciente para a base DREAMS Sleep
# Spindles. Usar tabela gerada pela função generate_eeg_data.
# Para cada sinal, a função aplica 5 filtros passa-faixa, gerando 5 novos sinais
# filtrados nas frequências:
#   delta: 0 a 4 Hz
#   theta: 5 a 7 Hz
#   alpha: 8 a 12 Hz
#   sigma: 13 a 15 Hz
#   beta: 16 a 30 Hz
#
# Parâmetros:
#   1. eeg_data - Matriz contendo os 3 sinais eeg
#
# Retorno:
#   Lista com os sub sinais representando os sinais filtrados nas 5 faixas de 
# frequência
#   d$XX$YY | XX = {s1, s2, s3}, YY = {delta, theta, alpha, sigma, beta}
#
#   d$s1$delta[,1] <- t
#   d$s1$delta[,2] <- sinal filtrado (0 a 4 Hz)
#
#   d$s1$theta[,1] <- t
#   d$s1$theta[,2] <- sinal filtrado (5 a 7 Hz)
#
# Exemplo:
#   sleep1 <- separate_sleep_eeg_band(d1)
#   sleep2 <- separate_sleep_eeg_band(d2)
#   sleep3 <- separate_sleep_eeg_band(d3)
#   sleep4 <- separate_sleep_eeg_band(d4)
#   sleep5 <- separate_sleep_eeg_band(d5)
#   sleep6 <- separate_sleep_eeg_band(d6)
#   sleep7 <- separate_sleep_eeg_band(d7)
#   sleep8 <- separate_sleep_eeg_band(d8)
##################################################################################
separate_sleep_eeg_band <- function(eeg_data){
  require(astrochron)
  
  separated_eeg <- NULL
  
  separated_eeg$s1$delta <- as.matrix(bandpass(eeg_data[,c(1,2)], 
                                               fhigh = 4, 
                                               genplot = FALSE, win = 2))
  separated_eeg$s2$delta <- as.matrix(bandpass(eeg_data[,c(1,3)], 
                                               fhigh = 4, 
                                               genplot = FALSE, win = 2))
  separated_eeg$s3$delta <- as.matrix(bandpass(eeg_data[,c(1,4)], 
                                               fhigh = 4, 
                                               genplot = FALSE, win = 2))

  separated_eeg$s1$theta <- as.matrix(bandpass(eeg_data[,c(1,2)], 
                                               flow = 5, fhigh = 7, 
                                               genplot = FALSE, win = 2))
  separated_eeg$s2$theta <- as.matrix(bandpass(eeg_data[,c(1,3)], 
                                               flow = 5, fhigh = 7, 
                                               genplot = FALSE, win = 2))
  separated_eeg$s3$theta <- as.matrix(bandpass(eeg_data[,c(1,4)], 
                                               flow = 5, fhigh = 7, 
                                               genplot = FALSE, win = 2))
  
  separated_eeg$s1$alpha <- as.matrix(bandpass(eeg_data[,c(1,2)], 
                                               flow = 8, fhigh = 12, 
                                               genplot = FALSE, win = 2))
  separated_eeg$s2$alpha <- as.matrix(bandpass(eeg_data[,c(1,3)], 
                                               flow = 8, fhigh = 12, 
                                               genplot = FALSE, win = 2))
  separated_eeg$s3$alpha <- as.matrix(bandpass(eeg_data[,c(1,4)], 
                                               flow = 8, fhigh = 12, 
                                               genplot = FALSE, win = 2))
  
  separated_eeg$s1$sigma <- as.matrix(bandpass(eeg_data[,c(1,2)], 
                                               flow = 13, fhigh = 15, 
                                               genplot = FALSE, win = 2))
  separated_eeg$s2$sigma <- as.matrix(bandpass(eeg_data[,c(1,3)], 
                                               flow = 13, fhigh = 15, 
                                               genplot = FALSE, win = 2))
  separated_eeg$s3$sigma <- as.matrix(bandpass(eeg_data[,c(1,4)], 
                                               flow = 13, fhigh = 15, 
                                               genplot = FALSE, win = 2))
  
  separated_eeg$s1$beta <- as.matrix(bandpass(eeg_data[,c(1,2)], 
                                              flow = 16, 
                                              fhigh = 30, genplot = FALSE, win = 2))
  separated_eeg$s2$beta <- as.matrix(bandpass(eeg_data[,c(1,3)], 
                                              flow = 16, fhigh = 30, 
                                              genplot = FALSE, win = 2))
  separated_eeg$s3$beta <- as.matrix(bandpass(eeg_data[,c(1,4)], 
                                              flow = 16, fhigh = 30, 
                                              genplot = FALSE, win = 2))
  
  return(separated_eeg)
}

##################################################################################
# curve_length(x,y)
#
# Calcula o comprimento de uma curva representada em coordenadas cartesianas
#
# Parâmetros
#   1. x - Coordenadas x da curva
#   2. y - Coordenadas y da curva
#
# Retorna:
#   Valor numérico do comprimento
##################################################################################
curve_length <- function(x,y){
  len = 0
  for(i in 2:length(x)){
    len = len + ((x[i]-x[i-1])^2+(y[i]-y[i-1])^2)^.5
  }
  return(len)
}

##################################################################################
# generate_features(d, freq, n_dwt, seg_len, nrm, excerpt)
#
# Cria uma matriz de features do sinal EEG principal.
# Primeiro, decompõe o sinal usando a transformada discreta de wavelets em n
# níveis. Para cada intervalo de tempo x, extrai de cada sub sinal média da
# amplitude, desvio padrão, energia, simetria e comprimento de curva.
#
# Parâmetros:
#   1. d      - Matriz que contêm o sinal EEG. (generate_eeg_data)
#   2. freq   - Frequência de amostragem do sinal, em Hz
#   3. n_dwt  - Níveis da transformada de Wavelets
#   4. seg_len- Duração dos segmentos de tempo, em segundos
#   5. nrm    - Deseja normalizar os dados?
#   6. excerpt- Número do paciente
#
# Retorna:
#   Matriz de features.
#   Gera arquivo em disco: wav_exX.csv | X = {1,2,3,4,5,6,7,8}
#   Colunas da tabela:
#     Sejam os sub-sinais: {D1, D2, ..., DN}
#     e os segmentos de tempo: {seg1, seg2, ..., segM}
#
#     Tabela:
#       seg1: | média(D1) | SD(D1) | Enrg(D1) | CurvLen(D1) | Sim(D1) | média (D2) | ... 
#       seg2: | média(D1) | SD(D1) | Enrg(D1) | CurvLen(D1) | Sim(D1) | média (D2) | ... 
#       ...
#
# Exemplo:
#   freq = 256
#   n_dwt = 5
#   seg_len = 2
#
#   f1 <- generate_features(d1, freq, n_dwt, seg_len, TRUE, 1)
#   f2 <- generate_features(d2, freq, n_dwt, seg_len, TRUE, 2)
#   f3 <- generate_features(d3, freq, n_dwt, seg_len, TRUE, 3)
#   f4 <- generate_features(d4, freq, n_dwt, seg_len, TRUE, 4)
#   f5 <- generate_features(d5, freq, n_dwt, seg_len, TRUE, 5)
#   f6 <- generate_features(d6, freq, n_dwt, seg_len, TRUE, 6)
#   f7 <- generate_features(d7, freq, n_dwt, seg_len, TRUE, 7)
#   f8 <- generate_features(d8, freq, n_dwt, seg_len, TRUE, 8)
##################################################################################
generate_features <- function(d, freq, n_dwt, seg_len, nrm, excerpt){
  require(psd)
  require(waveslim)
  require(e1071)
  require(ppls)
  
  n_seg = length(d[,1])/(seg_len*freq)
  inc = length(d[,1])/n_seg
  
  wd = 0
  dt <- matrix(nrow = n_seg, ncol = (n_dwt)*5 + 1)
  
  for(i in 0:(n_seg-1)){
    if(i/10 == as.integer(i/10)){
      print(paste(round(i*100/n_seg,2), '%', sep = ' '))
    }
    wd <- dwt(d[(i*inc+1):((i+1)*inc),2], n.levels = n_dwt)
    
    for(j in 1:n_dwt){
      media = mean(wd[[j]])
      desvP = sd(wd[[j]])
      
      max_t = max(d[(i*inc+1):((i+1)*inc),1])
      min_t = min(d[(i*inc+1):((i+1)*inc),1])
      
      if(j == length(wd)){
        spct <- pspectrum(wd[[j]], niter = 20, x.frqsamp = freq/(2^(length(wd) - 1)), verbose = FALSE)
        curvlen = curve_length(seq(min_t,max_t, ((2^j)-1)/freq), wd[[j]])
      }else{
        spct <- pspectrum(wd[[j]], niter = 20, x.frqsamp = freq/2^j, verbose = FALSE)
        curvlen = curve_length(seq(min_t,max_t, (2^j)/freq), wd[[j]])
      }
      energ = mean(spct$spec)
      skewn = skewness(wd[[j]])
      
      dt[(i+1),(j-1)*5 + 1] <- media
      dt[(i+1),(j-1)*5 + 2] <- desvP
      dt[(i+1),(j-1)*5 + 3] <- energ
      dt[(i+1),(j-1)*5 + 4] <- curvlen
      dt[(i+1),(j-1)*5 + 5] <- skewn
    }
    
    if(sum(d[(i*inc+1):((i+1)*inc),length(d[1,])]) > 0){
      dt[(i+1),(n_dwt)*5 + 1] <- 1
    }else{
      dt[(i+1),(n_dwt)*5 + 1] <- 0
    }
  }
  
  if(nrm){
    for(i in 1:(n_dwt*5)){
      dt[,i] <- normalize.vector(dt[,i])
    }
  }
  write.table(dt, file = paste("wav1_seg_ex", excerpt, ".csv", sep = ""), quote = FALSE, sep = ',', row.names = FALSE, col.names = FALSE)
  return(dt)
}

##################################################################################
# generate_features_all(d, freq, n_dwt, seg_len, nrm, excerpt)
#
# Cria uma matriz de features do sinal EEG principal.
# Primeiro, decompõe o sinal usando a transformada discreta de wavelets em n
# níveis. Para cada intervalo de tempo x, extrai de cada sub sinal média da
# amplitude, desvio padrão, energia, simetria e comprimento de curva.
#
# Parâmetros:
#   1. d      - Matriz que contêm o sinal EEG. (generate_eeg_data)
#   2. freq   - Frequência de amostragem do sinal, em Hz
#   3. n_dwt  - Níveis da transformada de Wavelets
#   4. seg_len- Duração dos segmentos de tempo, em segundos
#   5. nrm    - Deseja normalizar os dados?
#   6. excerpt- Número do paciente
#
# Retorna:
#   Matriz de features.
#   Gera arquivo em disco: wav_all_exX.csv | X = {1,2,3,4,5,6,7,8}#   
#   Colunas da tabela:
#     Sejam os sinais C, O, F,
#     os sub-sinais: {D1, D2, ..., DN},
#     e os segmentos de tempo: {seg1, seg2, ..., segM}
#
#     Tabela:
#       seg1: | média(D1_C) | SD(D1_C) | Enrg(D1_C) | CurvLen(D1_C) | Sim(D1_C) | ...
#             | média(DN_C) | ...  ... | Sim(DN_C)  | média(D1_O)   | ...
#       seg2: | média(D1) | SD(D1) | Enrg(D1) | CurvLen(D1) | Sim(D1) | média (D2) | ... 
#       ...
#
# Exemplo:
#   freq = 256
#   n_dwt = 5
#   seg_len = 2
#
#   f1 <- generate_features_all(d1, freq, n_dwt, seg_len, TRUE, 1)
#   f2 <- generate_features_all(d2, freq, n_dwt, seg_len, TRUE, 2)
#   f3 <- generate_features_all(d3, freq, n_dwt, seg_len, TRUE, 3)
#   f4 <- generate_features_all(d4, freq, n_dwt, seg_len, TRUE, 4)
#   f5 <- generate_features_all(d5, freq, n_dwt, seg_len, TRUE, 5)
#   f6 <- generate_features_all(d6, freq, n_dwt, seg_len, TRUE, 6)
#   f7 <- generate_features_all(d7, freq, n_dwt, seg_len, TRUE, 7)
#   f8 <- generate_features_all(d8, freq, n_dwt, seg_len, TRUE, 8)
##################################################################################
generate_features_all <- function(d, freq, n_dwt, seg_len, nrm, excerpt){
  require(psd)
  require(waveslim)
  require(e1071)
  require(ppls)
  
  n_seg = length(d[,1])/(seg_len*freq)
  inc = length(d[,1])/n_seg
  
  wd = 0
  dt <- matrix(nrow = n_seg, ncol = (n_dwt)*5*3 + 1)
  
  for(k in 0:2){
    for(i in 0:(n_seg-1)){
      if(i/50 == as.integer(i/50)){
        print(paste('Sinal', (k+1), '->', round(i*100/n_seg,1), '%', sep = ' '))
        }
      wd <- dwt(d[(i*inc+1):((i+1)*inc),2+k], n.levels = n_dwt)
      
      for(j in 1:n_dwt){
        media = mean(wd[[j]])
        desvP = sd(wd[[j]])
      
        max_t = max(d[(i*inc+1):((i+1)*inc),1])
        min_t = min(d[(i*inc+1):((i+1)*inc),1])
        
        if(j == length(wd)){
          spct <- pspectrum(wd[[j]], niter = 20, 
                            x.frqsamp = freq/(2^(length(wd) - 1)), verbose = FALSE)
          curvlen = curve_length(seq(min_t,max_t, ((2^j)-1)/freq), wd[[j]])
        }else{
          spct <- pspectrum(wd[[j]], niter = 20, 
                            x.frqsamp = freq/2^j, verbose = FALSE)
          curvlen = curve_length(seq(min_t,max_t, (2^j)/freq), wd[[j]])
        }
        energ = mean(spct$spec)
        skewn = skewness(wd[[j]])
        
        dt[(i+1),(j-1)*5 + 1 + k*25] <- media
        dt[(i+1),(j-1)*5 + 2 + k*25] <- desvP
        dt[(i+1),(j-1)*5 + 3 + k*25] <- energ
        dt[(i+1),(j-1)*5 + 4 + k*25] <- curvlen
        dt[(i+1),(j-1)*5 + 5 + k*25] <- skewn
      }
    
      if(sum(d[(i*inc+1):((i+1)*inc),length(d[1,])]) > 0){
        dt[(i+1),(n_dwt)*5*3 + 1] <- 1
      }else{
        dt[(i+1),(n_dwt)*5*3 + 1] <- 0
      }
    }
  }
    if(nrm){
      for(i in 1:(n_dwt*5*3)){
        dt[,i] <- normalize.vector(dt[,i])
      }
    }
    write.table(dt, file = paste("wav1_all_seg_ex", excerpt, ".csv", sep = ""), quote = FALSE, sep = ',', row.names = FALSE, col.names = FALSE)
    return(dt)
}

##################################################################################
# plot_PCA(f, features, limit, title)
#
# Cria gŕaficos mostrando a variância e a porcentagem de variância de cada 
# componente principal.
# 
# Parâmetros:
#   1. f        - Tabela com as features
#   2. features - Vetor com os índices das colunas desejadas
#   3. limit    - Valor (0 a 1) que indica a porcentagem de variância desejada
#   4. title    - Subtítulo para o gráfico
#
# Retorna:
#   Gráfico em tela
#   Número de componentes que satisfazem o limite.
#
# Exemplo:
#   n_comp <- plot_PCA(f, c(1:75), .95, "Paciente 1")
#
##################################################################################
plot_PCA <- function(f, features, limit, title){
  prin_comp <- prcomp(f[,features], scale. = T, center = T)
  std_dev <- prin_comp$sdev
  pr_var <- std_dev^2
  prop_varex <- pr_var/sum(pr_var)
  
  par(mfrow = c(1,2), oma = c(0,0,1,0))
  plot(prop_varex, xlab = "Principal Component",
       ylab = "Proportion of Variance Explained",
       type = "b", pch = 19)

  plot(cumsum(prop_varex), xlab = "Principal Component",
       ylab = "Cumulative Proportion of Variance Explained",
       type = "b", pch = 19)
  
  abline(h = limit, col = 'red', lty = 2, lwd = 2)
  n_pca = which(cumsum(prop_varex) >= limit)
  abline(v = min(n_pca), col = 'blue', lwd = 2)
  text(75, min(cumsum(prop_varex)+.05), 
       paste('Components for ', limit*100, '% \n of Variance: ', 
             min(n_pca), sep = ''), 
       pos = 2)
  
  mtext(paste('Análise da Componente Principal\n', title, sep = ''), outer = TRUE, cex = 1.2,side = 3, line = -2)
  return(min(n_pca))
}

##################################################################################
# data_PCA(f, features, n_comp, excerpt)
#
# Gera um arquivo csv com as n_comp componentes principais desejadas das features
# contidas na tabela f.
# 
# Parâmetros:
#   1. f        - Tabela com as features
#   2. features - Vetor com os índices das colunas desejadas
#   3. n_comp   - Número de componentes desejadas
#   4. excerpt  - Número do paciente
#
# Retorna:
#   Grava em disco o arquivo csv
#
# Exemplo:
#   data_PCA(f1, c(1:75), n_comp, "Paciente 1")
#
##################################################################################
data_PCA <- function(f, features, n_comp, excerpt){
  prin_comp <- prcomp(f[,features], scale. = T, center = T)
  temp = as.data.frame(prin_comp$x[,1:n_comp])
  temp[paste("PC",(n_comp+1),sep = "")] <- f1[,76]
  write.table(temp, file = paste("pca1_ex", excerpt, ".csv", sep = ""), quote = FALSE, sep = ',', row.names = FALSE, col.names = FALSE)
}

##################################################################################
generate_eeg_data_all_excerpts <- function(freq_out){
  d <- NULL
  d$d1 <- generate_eeg_data(1,1,100,freq_out)
  d$d2 <- generate_eeg_data(2,0,200,freq_out)
  d$d3 <- generate_eeg_data(3,1,50,freq_out)
  d$d4 <- generate_eeg_data(4,0,200,freq_out)
  d$d5 <- generate_eeg_data(5,0,200,freq_out)
  d$d6 <- generate_eeg_data(6,0,200,freq_out)
  d$d7 <- generate_eeg_data(7,0,200,freq_out)
  d$d8 <- generate_eeg_data(8,0,200,freq_out)
  
  return(d)
}

generate_features_all_excerpts <- function(d, freq, n_dwt, seg_len){
    f <- NULL
    f$f1 <- generate_features_all(d$d1, freq, n_dwt, seg_len, TRUE, 1)
    f$f2 <- generate_features_all(d$d2, freq, n_dwt, seg_len, TRUE, 2)
    f$f3 <- generate_features_all(d$d3, freq, n_dwt, seg_len, TRUE, 3)
    f$f4 <- generate_features_all(d$d4, freq, n_dwt, seg_len, TRUE, 4)
    f$f5 <- generate_features_all(d$d5, freq, n_dwt, seg_len, TRUE, 5)
    f$f6 <- generate_features_all(d$d6, freq, n_dwt, seg_len, TRUE, 6)
    f$f7 <- generate_features_all(d$d7, freq, n_dwt, seg_len, TRUE, 7)
    f$f8 <- generate_features_all(d$d8, freq, n_dwt, seg_len, TRUE, 8)
    return(f)
}

plot_PCA_all_excerpts <- function(f, comps, limit, name){
  n_comp1 <- plot_PCA(f$f1, comps, limit, paste(name, "1", sep = " "))
  print("Next?")
  scan()
  n_comp2 <- plot_PCA(f$f2, comps, limit, paste(name, "2", sep = " "))
  print("Next?")
  scan()
  n_comp3 <- plot_PCA(f$f3, comps, limit, paste(name, "3", sep = " "))
  print("Next?")
  scan()
  n_comp4 <- plot_PCA(f$f4, comps, limit, paste(name, "4", sep = " "))
  print("Next?")
  scan()
  n_comp5 <- plot_PCA(f$f5, comps, limit, paste(name, "5", sep = " "))
  print("Next?")
  scan()
  n_comp6 <- plot_PCA(f$f6, comps, limit, paste(name, "6", sep = " "))
  print("Next?")
  scan()
  n_comp7 <- plot_PCA(f$f7, comps, limit, paste(name, "7", sep = " "))
  print("Next?")
  scan()
  n_comp8 <- plot_PCA(f$f8, comps, limit, paste(name, "8", sep = " "))
  
  n_comps <- c(n_comp1,n_comp2,n_comp3,n_comp4,n_comp5,n_comp6,n_comp7,n_comp8)
  return(n_comps)
}

data_PCA_all_excerpts <- function(f, comps, n_comp){
  data_PCA(f$f1, comps, n_comp, 1)
  data_PCA(f$f2, comps, n_comp, 2)
  data_PCA(f$f3, comps, n_comp, 3)
  data_PCA(f$f4, comps, n_comp, 4)
  data_PCA(f$f5, comps, n_comp, 5)
  data_PCA(f$f6, comps, n_comp, 6)
  data_PCA(f$f7, comps, n_comp, 7)
  data_PCA(f$f8, comps, n_comp, 8)
}

create_database <- function(){
  freq_out = 256
  n_dwt = 5
  seg_len = 2
  comps = c(1:75)
  limit = .95
  name = "Paciente"
  
  d <- generate_eeg_data_all_excerpts(freq_out)
  f <- generate_features_all_excerpts(d, freq_out, n_dwt, seg_len)
  n_comps <- plot_PCA_all_excerpts <- function(f, comps, limit, name)
  
  n = max(n_comps)
  data_PCA_all_excerpts(f, comps, n)
}

##################################################################################

predict_to_time <- function(pred){
  vec <- matrix(nrow = sum(pred), ncol = 1)
  count = 1
  for(i in 1:length(pred)){
    if(pred[i]){
      vec[count,1] = i*2
      count = count + 1
    }
  }
  
  ret <- NULL
  ret$`[Spindles/C3-A1]` <- vec[,1]
  ret$`[Spindles/CZ-A1]` <- vec[,1]
  ret$Dur <- rep(2,length(vec))

  return(ret)
}

