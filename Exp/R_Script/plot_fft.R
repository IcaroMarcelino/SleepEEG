plot_fft <- function(t, y, title_plot, xmin, xmax){
  FFT <- fft(y)
  magn <- Mod(FFT)
  phase <- Arg(FFT)
  #plot(magn, type="l")
  magn.1 <- 20*log10(magn[1:(length(FFT)/2)])
  phase.1 <- phase[1:(length(FFT)/2)]
  
  x.axis <- 1:length(magn.1)/(max(t)-min(t))
  teste <- matrix(nrow = length(x.axis), ncol = 2)
  teste[,1]<-x.axis
  teste[,2]<-magn.1
  plot(teste, col = 'black', type="l", xlim = c(xmin,xmax), xlab = "Frequency (Hz)", ylab = "Magnitude (dB)", main = title_plot)
  #plot(teste,type="l", xlab = "Frequency (Hz)", ylab = "Phase (°)", main = title_plot)
  
  teste[which(teste[,2] == max(magn.1)),]
}

# excerpt: Excerpt numeber
# op:      1 for channel C3-A1, 0 for channel CZ-A1
# leg_pos: Legend position
plot_ch_central <- function(excerpt, op, leg_pos, ret){
  library(edf)
  library(readr)
  
  Automatic_detection_excerpt1 <- ret
  Visual_scoring1_excerpt1 <- read_table2(paste("Visual_scoring1_excerpt", excerpt, ".txt", sep = ""))
  
  if(excerpt != 7 && excerpt != 8 ){
    Visual_scoring2_excerpt1 <- read_table2(paste("Visual_scoring2_excerpt", excerpt, ".txt", sep = ""))
  }
  excerpt1 <- read.edf(paste("excerpt", excerpt, ".edf", sep = ""))
  
  if(op == 1){
    par(mar=c(5,4,4,5)+.1)
    plot(excerpt1$signal$C3_A1$t,excerpt1$signal$C3_A1$data, type = 'l', xlab = "Tempo (s)", ylab = "Amplitude (uV)", main = paste("Fusos do Sono \n Paciente", excerpt,"- Canal C3-A1"), ylim = c(-300,300))
    segments(Automatic_detection_excerpt1$`[KComplexes/C3-A1]`, 0, Automatic_detection_excerpt1$`[KComplexes/C3-A1]`+Automatic_detection_excerpt1$Dur, 0, col = "orange", lwd = 2)
    segments(Visual_scoring1_excerpt1$`[vis1_KComplexes/C3-A1]`, 10, Visual_scoring1_excerpt1$`[vis1_KComplexes/C3-A1]`+Visual_scoring1_excerpt1$Dur, 10, col = "green", lwd = 2)
    
    if(excerpt != 7 && excerpt != 8 ){
      segments(Visual_scoring2_excerpt1$`[vis2_KComplexes/C3-A1]`, -10, Visual_scoring2_excerpt1$`[vis2_KComplexes/C3-A1]`+Visual_scoring2_excerpt1$Dur, -10, col = "blue", lwd = 2)
    }
  }
  else{
    par(mar=c(5,4,4,5)+.1)
    plot(excerpt1$signal$CZ_A1$t,excerpt1$signal$CZ_A1$data, xlim = c(0,1800), type = 'l', xlab = "Tempo (s)", ylab = "Amplitude (uV)", main = paste("Complexos K \n Paciente", excerpt,"- Canal CZ-A1"), ylim = c(-300,300))
    segments(Automatic_detection_excerpt1$`[Spindles/CZ-A1]`, 0, Automatic_detection_excerpt1$`[Spindles/CZ-A1]`+Automatic_detection_excerpt1$Dur, 0, col = "orange", lwd = 2)
    segments(Visual_scoring2_excerpt1$`[Spindles/CZ-A1]`, -10, Visual_scoring2_excerpt1$`[Spindles/CZ-A1]`+Visual_scoring2_excerpt1$Dur, -10, col = "blue", lwd = 2)
    segments(Visual_scoring2_excerpt1$`[vis2_Kcomplexes/CZ-A1]`, -10, Visual_scoring2_excerpt1$`[vis2_Kcomplexes/CZ-A1]`+Visual_scoring2_excerpt1$Dur, -10, col = "blue", lwd = 2)
    legend(leg_pos,fill=c("orange", "blue"),legend=c("Automático", "Especialista 1 U Especialista 2"), inset=.02, title="Detecção de Complexos K",bty = "n")
    
    segments(Visual_scoring1_excerpt1$`[vis1_Spindles/CZ-A1]`, 10, Visual_scoring1_excerpt1$`[vis1_Spindles/CZ-A1]`+Visual_scoring1_excerpt1$Dur, 10, col = "green", lwd = 2)
    segments(Visual_scoring1_excerpt1$`[vis1_Kcomplexes/CZ-A1]`, 10, Visual_scoring1_excerpt1$`[vis1_Kcomplexes/CZ-A1]`+Visual_scoring1_excerpt1$Dur, 10, col = "green", lwd = 2)
    
    if(excerpt != 7 && excerpt != 8 ){
      segments(Visual_scoring2_excerpt1$`[vis2_Spindles/CZ-A1]`, -10, Visual_scoring2_excerpt1$`[vis2_Spindles/CZ-A1]`+Visual_scoring2_excerpt1$Dur, -10, col = "blue", lwd = 2)
    }
  }
  
  par(new=TRUE)
  #lines(excerpt1$signal$hypnogram$t,excerpt1$signal$hypnogram$data, col = "red", lwd = 2, xaxt="n",yaxt="n",xlab="",ylab="", type = "l")
  plot(excerpt1$signal$hypnogram$t,excerpt1$signal$hypnogram$data, col = "red", lwd = 2, xaxt="n",yaxt="n",xlab="",ylab="", type = "l")
  axis(4, col.axis="red", at = c(5,4,3,2,1,0), labels = c("Acordado", "REM", "S1", "S2", "S3","S4"), las =2)
  mtext("Estágio do Sono",side=4,line=3, col = "red")
  #legend(leg_pos,fill=c("orange","green", "blue"),legend=c("Automatic","Expert 1", "Expert 2"), inset=.02, title="Spindle Detection",bty = "n")
  legend(leg_pos,fill=c("orange","green", "blue"),legend=c("Automático", "Especialista 1", "Especialista 2"), inset=.02, title="Detecção de Complexos K",bty = "n")
}

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
    par(mar=c(5,4,4,5)+.1)
    plot(excerpt1$signal$C3_A1$t,excerpt1$signal$C3_A1$data, xlim = c(540,545), type = 'l', xlab = "Time (s)", ylab = "Amplitude (uV)", main = paste("Sleep KComplexes \nExcerpt", excerpt,"- Channel C3-A1"), ylim = c(-100,100))
    #segments(Automatic_detection_excerpt1$`[KComplexes/C3-A1]`, 0, Automatic_detection_excerpt1$`[KComplexes/C3-A1]`+Automatic_detection_excerpt1$Dur, 0, col = "orange", lwd = 2)
    segments(Visual_scoring1_excerpt1$`[vis1_KComplexes/C3-A1]`, 5, Visual_scoring1_excerpt1$`[vis1_KComplexes/C3-A1]`+Visual_scoring1_excerpt1$Dur, 5, col = "green", lwd = 2)
    
    if(excerpt != 7 && excerpt != 8 ){
      segments(Visual_scoring2_excerpt1$`[vis2_KComplexes/C3-A1]`, -5, Visual_scoring2_excerpt1$`[vis2_KComplexes/C3-A1]`+Visual_scoring2_excerpt1$Dur, -5, col = "blue", lwd = 2)
    }
  }
  else{
    par(mar=c(5,4,4,5)+.1)
    plot(excerpt1$signal$CZ_A1$t,excerpt1$signal$CZ_A1$data, type = 'l', xlab = "Time (s)", ylab = "Amplitude (uV)", main = paste("Sleep KComplexes \n Excerpt", excerpt,"- Channel CZ-A1"), ylim = c(-300,300))
    #segments(Automatic_detection_excerpt1$`[KComplexes/CZ-A1]`, 0, Automatic_detection_excerpt1$`[KComplexes/CZ-A1]`+Automatic_detection_excerpt1$Dur, 0, col = "orange", lwd = 2)
    segments(Visual_scoring1_excerpt1$`[vis1_KComplexes/CZ-A1]`, 20, Visual_scoring1_excerpt1$`[vis1_KComplexes/CZ-A1]`+Visual_scoring1_excerpt1$Dur, 20, col = "green", lwd = 2)
    
    if(excerpt != 7 && excerpt != 8 ){
      segments(Visual_scoring2_excerpt1$`[vis2_KComplexes/CZ-A1]`, -20, Visual_scoring2_excerpt1$`[vis2_KComplexes/CZ-A1]`+Visual_scoring2_excerpt1$Dur, -20, col = "blue", lwd = 2)
    }
  }
  
  par(new=TRUE)
  #lines(excerpt1$signal$hypnogram$t,excerpt1$signal$hypnogram$data*50, col = "red", lwd = 2)
  plot(excerpt1$signal$hypnogram$t,excerpt1$signal$hypnogram$data, xlim = c(540,545), col = "red", lwd = 2, xaxt="n",yaxt="n",xlab="",ylab="", type = "l")
  axis(4, col.axis="red", at = c(5,4,3,2,1,0), labels = c("Wake", "REM", "S1", "S2", "S3","S4"), las =2)
  mtext("Sleep Stage",side=4,line=3, col = "red")
  #legend(leg_pos,fill=c("orange","green", "blue"),legend=c("Automatic","Expert 1", "Expert 2"), inset=.02, title="Spindle Detection",bty = "n")
  legend(leg_pos,fill=c("green", "blue"),legend=c("Expert 1", "Expert 2"), inset=.02, title="Sleep Spindle Detection",bty = "n")
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

d1 <- generate_data(1,1,100)
d2 <- generate_data(2,0,200)
d3 <- generate_data(3,1,50)
d4 <- generate_data(4,0,200)
d5 <- generate_data(5,0,200)
d6 <- generate_data(6,0,200)
d7 <- generate_data(7,0,200)
d8 <- generate_data(8,0,200)

plot_ch_central(1, 1, "topright")
plot_ch_central(2, 0, "topleft")
plot_ch_central(3, 1, "bottomleft")
plot_ch_central(4, 0, "bottomleft")
plot_ch_central(5, 0, "topright")
plot_ch_central(6, 0, "topright")
plot_ch_central(7, 0, "topright")
plot_ch_central(8, 0, "bottomright")
