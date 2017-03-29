##############################
## Datasets of Mikas Vengris, Denitsa Grancharova 
## and Rienk van Grondelle 
##############################

##############################
## load TIMP
##############################

require(TIMP)

##############################
## READ IN PSI 1
##############################

denS4<-readData("psi_1.txt")

##############################
## PREPROCESS PSI 1
##############################

denS4<-preProcess(data = denS4, scalx2 = c(3.78, 643.5))

##############################
## READ IN PSI 2
##############################

denS5<-readData("psi_2.txt")

##############################
## PREPROCESS PSI 2
##############################

denS5<-preProcess(data = denS5, scalx2 = c(3.78, 643.5))

##############################
## DEFINE INITIAL MODEL
##############################

model1<- initModel(mod_type = "kin", 
kinpar= c(7.9, 1.08, 0.129, .0225, .00156) , 
irfpar=c( -.1018, 0.0434), 
disptau=FALSE, dispmu=TRUE, parmu = list(c(.230)), 
lambdac = 650,
seqmod=TRUE,
positivepar=c("kinpar"),
title="S4", 
cohspec = list( type = "irf"))

##############################
## FIT INITIAL MODEL
##############################

denRes<-fitModel(data=list(denS4, denS5), list(model1), 
opt=kinopt(iter=5, superimpose = c(1,2), divdrel = TRUE, linrange = .2,
makeps = "den1", selectedtraces = c(1,5,10), plotkinspec =TRUE, 
xlab = "time (ps)", ylab = "wavelength"))

##############################
## REFINE INITIAL MODEL, RE-FIT
## adding some per-dataset parameters 
##############################

denRes<-fitModel(data = list(denS4, denS5), list(model1),
modeldiffs = list(dscal = list(list(to=2,from=1,value=.457)), 
free = list(
list(what = "irfpar", ind = 1, dataset = 2, start=-.1932),
list(what = "kinpar", ind = 5, dataset = 2, start=.0004), 
list(what = "kinpar", ind = 4, dataset = 2, start= .0159)
)),
opt=kinopt(iter=5,superimpose = c(1,2), divdrel = TRUE, linrange = .2,
xlab = "time (ps)", ylab = "wavelength",
makeps = "den2", selectedtraces = c(1,5,10)))

##############################
## REFINE MODEL FURTHER AS NEW MODEL OBJECT 
##############################

model2 <- initModel(mod_type = "kin", 
kinpar= c(7.9, 1.08, 0.129, .0225, .00156), 
irfpar=c( -.1018, 0.0434), 
parmu = list(c(.230)), 
lambdac = 650,
seqmod=TRUE,
positivepar=c("kinpar", "coh"), 
cohspec = list( type = "seq", start = c(8000, 1800)))

##############################
## FIT NEW MODEL OBJECT
##############################

denRes<-fitModel(data = list(denS4, denS5), list(model2),
modeldiffs = list(dscal = list(list(to=2,from=1,value=.457)), 
free = list(
list(what = "irfpar", ind = 1, dataset = 2, start=-.1932),
list(what = "kinpar", ind = 5, dataset = 2, start=.0004), 
list(what = "kinpar", ind = 4, dataset = 2, start= .0159)
)),
opt=kinopt(iter=5,superimpose = c(1,2), divdrel = TRUE, linrange = .2,
makeps = "den3", selectedtraces = c(1,5,10), plotkinspec =TRUE, 
paropt=list(cex.main=1.2,cex.lab=1.2,cex.axis=1.2),
stderrclp = TRUE, kinspecerr=TRUE,
xlab = "time (ps)", ylab = "wavelength", 
breakdown = list(plot=c(643.50, 658.62, 677.52), 
superimpose=1:2 )))




