# excerpt: Excerpt numeber
# op:      1 for channel C3-A1, 0 for channel CZ-A1
# leg_pos: Legend position
plot_ch_central <- function(excerpt, op, leg_pos){
  library(edf)
  library(readr)
  
  Automatic_detection_excerpt1 <- read_table2(paste("Automatic_detection_excerpt", excerpt, ".txt", sep = ""))
  Visual_scoring1_excerpt1 <- read_table2(paste("Visual_scoring1_excerpt", excerpt, ".txt", sep = ""))
  
  if(excerpt != 7 && excerpt != 8 ){
    Visual_scoring2_excerpt1 <- read_table2(paste("Visual_scoring2_excerpt", excerpt, ".txt", sep = ""))
  }
  excerpt1 <- read.edf(paste("excerpt", excerpt, ".edf", sep = ""))
  
  if(op == 1){
   #par(mar=c(5,4,4,5)+.1)
    plot(excerpt1$signal$C3_A1$t,excerpt1$signal$C3_A1$data, type = 'l', xlab = "Tempo (s)", ylab = "Amplitude (uV)", main = paste("Fusos do Sono \n Paciente", excerpt,"- Canal C3-A1"), ylim = c(-300,300))
    #segments(Automatic_detection_excerpt1$`[Spindles/C3-A1]`, 0, Automatic_detection_excerpt1$`[Spindles/C3-A1]`+Automatic_detection_excerpt1$Dur, 0, col = "orange", lwd = 2)
    segments(Visual_scoring1_excerpt1$`[vis1_Spindles/C3-A1]`, 15, Visual_scoring1_excerpt1$`[vis1_Spindles/C3-A1]`+Visual_scoring1_excerpt1$Dur, 15, col = "green", lwd = 2)
    
    if(excerpt != 7 && excerpt != 8 ){
      segments(Visual_scoring2_excerpt1$`[vis2_Spindles/C3-A1]`, -15, Visual_scoring2_excerpt1$`[vis2_Spindles/C3-A1]`+Visual_scoring2_excerpt1$Dur, -15, col = "blue", lwd = 2)
    }
  }
  else{
    #par(mar=c(5,4,4,5)+.1)
    plot(excerpt1$signal$CZ_A1$t,excerpt1$signal$CZ_A1$data, type = 'l', xlab = "Tempo (s)", ylab = "Amplitude (uV)", main = paste("Fusos do Sono \n Paciente", excerpt,"- Canal CZ-A1"), ylim = c(-300,300))
    #segments(Automatic_detection_excerpt1$`[Spindles/CZ-A1]`, 0, Automatic_detection_excerpt1$`[Spindles/CZ-A1]`+Automatic_detection_excerpt1$Dur, 0, col = "orange", lwd = 2)
    segments(Visual_scoring1_excerpt1$`[vis1_Spindles/CZ-A1]`, 15, Visual_scoring1_excerpt1$`[vis1_Spindles/CZ-A1]`+Visual_scoring1_excerpt1$Dur, 15, col = "green", lwd = 2)
    
    if(excerpt != 7 && excerpt != 8 ){
      segments(Visual_scoring2_excerpt1$`[vis2_Spindles/CZ-A1]`, -15, Visual_scoring2_excerpt1$`[vis2_Spindles/CZ-A1]`+Visual_scoring2_excerpt1$Dur, -15, col = "blue", lwd = 2)
    }
  }
  
  #par(new=TRUE)
  #lines(excerpt1$signal$hypnogram$t,excerpt1$signal$hypnogram$data*50, col = "red", lwd = 2)
  #plot(excerpt1$signal$hypnogram$t,excerpt1$signal$hypnogram$data, col = "red", lwd = 2, xaxt="n",yaxt="n",xlab="",ylab="", type = "l")
  #axis(4, col.axis="red", at = c(5,4,3,2,1,0), labels = c("Acordado", "REM", "S1", "S2", "S3","S4"), las =2)
  #mtext("Estágio do Sono",side=4,line=3, col = "red")
  #legend(leg_pos,fill=c("orange","green", "blue"),legend=c("Automatic","Expert 1", "Expert 2"), inset=.02, title="Spindle Detection",bty = "n")
  legend(leg_pos,fill=c("green", "blue", "orange"),legend=c("Especialista 1", "Especialista 2", "Detecção Automática"), inset=.02, title="Detecção de Fusos do Sono",bty = "n")
}

generate_data <- function(excerpt, op, freq){
  library(edf)
  library(readr)
  library(ppls)
  
  auto <- read_table2(paste("Automatic_detection_excerpt", excerpt, ".txt", sep = ""), col_types = cols(col_double(), col_double()))
  exp1 <- read_table2(paste("Visual_scoring1_excerpt", excerpt, ".txt", sep = ""), col_types = cols(col_double(), col_double()))
  
  if(excerpt != 7 && excerpt != 8 ){
    exp2 <- read_table2(paste("Visual_scoring2_excerpt", excerpt, ".txt", sep = ""), col_types = cols(col_double(), col_double()))
  }
  excerpt_data <- read.edf(paste("excerpt", excerpt, ".edf", sep = ""))
  
  if(op == 1){
    d = matrix(nrow = length(excerpt_data$signal$C3_A1$t), ncol = 11)
    d[,1] = excerpt_data$signal$C3_A1$t
    d[,2] = normalize.vector(excerpt_data$signal$C3_A1$data)
  }else{
    d = matrix(nrow = length(excerpt_data$signal$CZ_A1$t), ncol = 11)
    d[,1] = excerpt_data$signal$CZ_A1$t
    d[,2] = normalize.vector(excerpt_data$signal$CZ_A1$data)
  }
  d[,3] = normalize.vector(excerpt_data$signal$FP1_A1$data)
  d[,4] = normalize.vector(excerpt_data$signal$O1_A1$data)
  d[,5] = 0
  d[,6] = 0
  d[,7] = 0
  d[,8] = 0
  d[,9] = 0
  d[,10] = 0
  d[,11] = 0
  
  for(i in 1:length(excerpt_data$signal$hypnogram$data)){
    d[c((i-1)*freq+1):(i*freq),(excerpt_data$signal$hypnogram$data[i]+5)] = 1
  }
  
  for(i in 1:length(auto[,1,1])){
    if(op == 1){
      a = which(excerpt_data$signal$C3_A1$t >= as.double(auto[i,1]))
      b = which(excerpt_data$signal$C3_A1$t <= as.double(auto[i,1]+as.double(auto[i,2])))
      c1 = intersect(a,b)
      
      a = which(excerpt_data$signal$C3_A1$t >= as.double(exp1[i,1]))
      b = which(excerpt_data$signal$C3_A1$t <= as.double(exp1[i,1]+as.double(exp1[i,2])))
      c2 = intersect(a,b)
      
      c = union(c1, c2)
      
      if(excerpt != 7 && excerpt != 8 ){
        a = which(excerpt_data$signal$C3_A1$t >= as.double(exp2[i,1]))
        b = which(excerpt_data$signal$C3_A1$t <= as.double(exp2[i,1]+as.double(exp2[i,2])))
        c3 = intersect(a,b)
        
        c3 = intersect(a,b)
        c = union(c, c3)
        
      }
      
      d[c, 11] = 1
    }else{
      a = which(excerpt_data$signal$CZ_A1$t >= as.double(auto[i,1]))
      b = which(excerpt_data$signal$CZ_A1$t <= as.double(auto[i,1]+as.double(auto[i,2])))
      c1 = intersect(a,b)
      
      a = which(excerpt_data$signal$CZ_A1$t >= as.double(exp1[i,1]))
      b = which(excerpt_data$signal$CZ_A1$t <= as.double(exp1[i,1]+as.double(exp1[i,2])))
      c2 = intersect(a,b)
      
      c = union(c1, c2)
      
      if(excerpt != 7 && excerpt != 8 ){
        a = which(excerpt_data$signal$CZ_A1$t >= as.double(exp2[i,1]))
        b = which(excerpt_data$signal$CZ_A1$t <= as.double(exp2[i,1]+as.double(exp2[i,2])))
        
        c3 = intersect(a,b)
        c = union(c, c3)
      }
      d[c, 11] = 1
    }
  }
  
  print(sum(d[,11])*100/length(d[,11]))
  #write.table(d, file = paste("ex", excerpt, ".csv", sep = ""), quote = FALSE, sep = ',', row.names = FALSE, col.names = FALSE)
  
  return(d)
}

generate_eeg_data <- function(excerpt, op, freq, freq_out){
  require(edf)
  require(readr)
  require(ppls)
  
  freq_factor = freq_out/freq
  
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
  c_exp1 = -1
  c_exp2 = -1
  c1 = -1
  c2 = -1
  
  for(i in 1:length(exp1[,1,1])){
    a = which(d[,1] >= as.double(exp1[i,1]))
    b = which(d[,1] <= as.double(exp1[i,1]+as.double(exp1[i,2])))
    c_exp1 = intersect(a,b)
      
    if(i == 1){
        c1 = c_exp1
      }
    c1 = union(c1, c_exp1)
  }
  
  c = c1
  if(excerpt != 7 && excerpt != 8 ){
    for(i in 1:length(exp2[,1,1])){
      a = which(d[,1] >= as.double(exp2[i,1]))
      b = which(d[,1] <= as.double(exp2[i,1]+as.double(exp2[i,2])))
      c_exp2 = intersect(a,b)
          
      if(i == 1){
        c2 = c_exp2
      }
      c2 = union(c2, c_exp2)
    }
    c = intersect(c1, c2)
  }
  d[c,5] = 1
  #print(sum(d[,5])*100/length(d[,5]))
  #write.table(d, file = paste("ex", excerpt, ".csv", sep = ""), quote = FALSE, sep = ',', row.names = FALSE, col.names = FALSE)
  
  return(d)
}

d1 <- generate_eeg_data(1,1,100,100)
d2 <- generate_eeg_data(2,0,200,200)
d3 <- generate_data(3,1,50)
d4 <- generate_data(4,0,200)
d5 <- generate_data(5,0,200)
d6 <- generate_data(6,0,200)
d7 <- generate_data(7,0,200)
d8 <- generate_data(8,0,200)

x <- d1[,1]
y <- d1[,5]*5

y[which(y == 0)] <- NA
points(x,y, col = 'red', pch = '.')

plot_ch_central(1, 1, "topright")
plot_ch_central(2, 0, "topleft")
plot_ch_central(3, 1, "bottomleft")
plot_ch_central(4, 0, "bottomleft")
plot_ch_central(5, 0, "topright")
plot_ch_central(6, 0, "topright")
plot_ch_central(7, 0, "topright")
plot_ch_central(8, 0, "bottomright")

a1 = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
a2 = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
       

pred = a2
ss = matrix(nrow = sum(pred), ncol = 2)
ss[,2] <- rep(2,sum(pred))
ss[,1] <- which(pred == 1)*2

segments(ss[,1], 0, ss[,1]+ss[,2], 0, col = "orange", lwd = 2)
