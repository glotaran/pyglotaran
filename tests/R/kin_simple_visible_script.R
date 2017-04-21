##############################
## Dataset and model of 
## Mariangela di Donato.
##############################

##############################
## load TIMP
##############################

require(TIMP)

##############################
## READ IN DATA
##############################

mdDat<-readData("cp8b.txt")

##############################
## Look at the attributes of the data object
##############################

slotNames(mdDat)

##############################
## The times in the data
##############################

mdDat@x 

##############################
## The number of timepoints in the data
##############################

mdDat@nt 

##############################
## The wavelengths in the data
##############################

mdDat@x2

##############################
## The number of wavelengths in the data
##############################

mdDat@nl

##############################
## PREPROCESS PSI 1
## estimate the baseline from timepoints 1-9, 
## and substract this baseline from data between wavelengths 1-256 
##############################

mdDat<-preProcess(data = mdDat, baselinelambda = c(1, 9, 1, 256))

##############################
## PREPROCESS PSI 1
## Select wavelengths 1-235 for analysis
##############################

mdDat<-preProcess(data = mdDat, sel_lambda = c(1, 235))


##############################
## DEFINE INITIAL MODEL
##############################

model1<- initModel(mod_type = "kin", 
kinpar= c(3.0, 0.5266452074, 0.8900437504E-01, 0.3096646687E-03 ) , 
irfpar=c(-0.213412169,0.66E-01), 
parmu = list(c(-0.4294245897, 0.663252115 )), 
lambdac = 670,
seqmod=TRUE,
positivepar=c("kinpar"), weightpar = list(c(-16, -.1, NA, NA, .5)),
title="visible example", 
cohspec = list( type = "irf"))

##############################
## FIT INITIAL MODEL
## plotting only every 20th trace
##############################

denRes<-fitModel(list(mdDat), list(model1), 
opt=kinopt(iter=33, linrange = .2,
makeps = "visible", xlab = "time (ps)", 
ylab = "wavelength", stderrclp = TRUE, 
plotkinspec = TRUE, kinspecerr = TRUE,
selectedtraces = seq(1,256,by=20)))
