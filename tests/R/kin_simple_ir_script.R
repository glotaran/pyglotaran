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

mdDat<-readData("cp432704.txt")

##############################
## PREPROCESS PSI 1
## estimate the baseline from timepoints 1-5, 
## and substract this baseline from data between wavelengths 1-32 
##############################

mdDat<-preProcess(data = mdDat, baselinelambda = c(1, 5, 1, 32))

##############################
## DEFINE INITIAL MODEL
##############################

model1<- initModel(mod_type = "kin", 
kinpar= c(1.0,  0.01, 0.005  ), irfpar=c(.003, .095), 
parmu = list(c(.073)), lambdac = 1670,
seqmod=TRUE,
positivepar = c("kinpar"), weightpar = list(c(-16, -.1, NA, NA, .1)),
makeps="MARI", title="MARI", 
cohspec = list( type = "irf"))

##############################
## FIT INITIAL MODEL
##############################

denRes<-fitModel(list(mdDat), list(model1), 
opt=kinopt(iter=5, linrange = .2,
makeps = "MARI", xlab = "time (ps)", 
ylab = "wavelength", plotkinspec = TRUE))

