plot_ch_central <- function(excerpt, op, leg_pos, pred){
  library(edf)
  library(readr)
  
  Automatic_detection_excerpt1 <- pred
  Visual_scoring1_excerpt1 <- read_table2(paste("Visual_scoring1_excerpt", excerpt, ".txt", sep = ""))
  
  if(excerpt != 7 && excerpt != 8 ){
    Visual_scoring2_excerpt1 <- read_table2(paste("Visual_scoring2_excerpt", excerpt, ".txt", sep = ""))
  }
  excerpt1 <- read.edf(paste("excerpt", excerpt, ".edf", sep = ""))
  
  if(op == 1){
    par(mar=c(5,4,4,5)+.1)
    plot(excerpt1$signal$C3_A1$t,excerpt1$signal$C3_A1$data, type = 'l', xlab = "Time (s)", ylab = "Amplitude (uV)", main = paste("Sleep Spindles \n Excerpt", excerpt,"- Channel C3-A1"), ylim = c(-300,300))
    segments(Automatic_detection_excerpt1$`[Spindles/C3-A1]`, 0, Automatic_detection_excerpt1$`[Spindles/C3-A1]`+Automatic_detection_excerpt1$Dur, 0, col = "orange", lwd = 2)
    segments(Visual_scoring1_excerpt1$`[vis1_Spindles/C3-A1]`, 20, Visual_scoring1_excerpt1$`[vis1_Spindles/C3-A1]`+Visual_scoring1_excerpt1$Dur, 20, col = "green", lwd = 2)
    
    if(excerpt != 7 && excerpt != 8 ){
      segments(Visual_scoring2_excerpt1$`[vis2_Spindles/C3-A1]`, -20, Visual_scoring2_excerpt1$`[vis2_Spindles/C3-A1]`+Visual_scoring2_excerpt1$Dur, -20, col = "blue", lwd = 2)
    }
  }
  else{
    par(mar=c(5,4,4,5)+.1)
    plot(excerpt1$signal$CZ_A1$t,excerpt1$signal$CZ_A1$data, type = 'l', xlab = "Time (s)", ylab = "Amplitude (uV)", main = paste("Sleep Spindles \n Excerpt", excerpt,"- Channel CZ-A1"), ylim = c(-300,300))
    segments(Automatic_detection_excerpt1$`[Spindles/CZ-A1]`, 0, Automatic_detection_excerpt1$`[Spindles/CZ-A1]`+Automatic_detection_excerpt1$Dur, 0, col = "orange", lwd = 2)
    segments(Visual_scoring1_excerpt1$`[vis1_Spindles/CZ-A1]`, 20, Visual_scoring1_excerpt1$`[vis1_Spindles/CZ-A1]`+Visual_scoring1_excerpt1$Dur, 20, col = "green", lwd = 2)
    
    if(excerpt != 7 && excerpt != 8 ){
      segments(Visual_scoring2_excerpt1$`[vis2_Spindles/CZ-A1]`, -20, Visual_scoring2_excerpt1$`[vis2_Spindles/CZ-A1]`+Visual_scoring2_excerpt1$Dur, -20, col = "blue", lwd = 2)
    }
  }
  
  par(new=TRUE)
  #lines(excerpt1$signal$hypnogram$t,excerpt1$signal$hypnogram$data*50, col = "red", lwd = 2)
  
  #plot(excerpt1$signal$hypnogram$t,excerpt1$signal$hypnogram$data, col = "red", lwd = 2, xaxt="n",yaxt="n",xlab="",ylab="", type = "l")
  #axis(4, col.axis="red", at = c(5,4,3,2,1,0), labels = c("Wake", "REM", "S1", "S2", "S3","S4"), las =2)
  #mtext("Sleep Stage",side=4,line=3, col = "red")
  legend(leg_pos,fill=c("orange","green", "blue"),legend=c("GP+KNN","Expert 1", "Expert 2"), inset=.02, title="Spindle Detection",bty = "n")
}
