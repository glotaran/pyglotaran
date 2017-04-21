##############################
## Dataset and model of
## Mariangela di Donato
##############################

##############################
## load TIMP
##############################

require(TIMP)

##############################
## Read in and preprocess (by baseline correction) 
## 4 datasets 
##############################

did1<-readData("0902.txt")
did1<-preProcess(data = did1, baselinelambda = c(3, 7, 1, 32))

did2<-readData("1013.txt")
did2<-preProcess(data = did2, baselinelambda = c(3, 9, 1, 32))

did3<-readData("124.txt")
did3<-preProcess(data = did3, baselinelambda = c(3, 7, 1, 32))

did4<-readData("1402.txt")
did4<-preProcess(data = did4, baselinelambda = c(3, 7, 1, 32))

##############################
## Make a list of the 4 datasets, to be 
## used for simulteneous analysis 
##############################

data <- list(did1, did2, did3, did4) 

##############################
## Initial model.  
##############################

model_allfree <- initModel(mod_type = "kin", 
kinpar=c(1.1, 0.05, 0.001), 
irfpar=c(.05, 0.495912E-01), 
lambdac = 1670,
parmu = list(-.0158119812957), 
seqmod=TRUE, positivepar=c("kinpar"), 
weightpar=list( c(-16,.3,1,2000,.1), c(NA,NA,1590,1612,.1)), 
cohspec = list( type = "irf"))

##############################
## First apply the initial model to each dataset separately. 
##############################

didRes1<-fitModel(list(did1), list(model_allfree), opt=kinopt(iter=2, 
makeps="d1", title = "Dataset 1, single analysis"))

didRes2<-fitModel(list(did2), list(model_allfree), opt=kinopt(iter=2,
makeps="d2", title = "Dataset 2, single analysis"))

didRes3<-fitModel(list(did3), list(model_allfree), opt=kinopt(iter=2,
makeps="d3", title = "Dataset 3, single analysis"))

didRes4<-fitModel(list(did4), list(model_allfree), opt=kinopt(iter=2,
makeps="d4", title = "Dataset 4, single analysis"))

##############################
## Apply the initial model to datasets 1 and 2 together. 
## Make the kinetic and irf parameters free to vary between 
## datasets, and add a spectral equality constraint between 
## datasets.  Also add a dataset scaling parameter. 
##############################

didRes<-fitModel(list(did1,did2), modspec = list(model_allfree), 
modeldiffs = list( dscal = list(list(to=1,from=2,value=1)),
change = list(
list(what="kinpar", spec= c(1.011, 0.05, 0.005), dataset=2),
list(what="irfpar", spec= c(.051, 0.496912E-01), dataset=2),
list(what="clpequspec", spec=list(list(to=3, from=3, low=100, high=1810, 
dataset=2)), dataset=1), 
list(what="clpequ", spec=c(1), dataset=1))), opt=kinopt(iter=5))

##############################
## Apply the initial model to datasets 3 and 4 together. 
## Make the kinetic and irf parameters free to vary between 
## datasets, and add a spectral equality constraint between 
## datasets.  Also add a dataset scaling parameter -- but this 
## time also add a higher order polynomial for dispersion to the 
## 2nd dataset (dataset 4) via the parmu parameterization.
##############################

didRes<-fitModel(list(did3,did4), list(model_allfree), 
modeldiffs = list( dscal = list(list(to=1,from=2,value=1)),
change = list(
list(what="kinpar", spec= c(1.0, 0.05, 0.005), dataset=2),
list(what="irfpar", spec= c(.05, 0.496912E-01), dataset=2),
list(what="clpequspec", spec=list(list(to=3, from=3, low=100, 
high=1810, dataset=1)), dataset=2),  
list(what="clpequ", spec=c(1), dataset=2), 
list(what = "parmu", spec=list(c(-.0158119812957)), dataset=2))), 
opt=kinopt(iter=1))

##############################
## Analyze all 4 datasets (previously put into the list "data") together.
## Make all the kinetic and IRF parameters free between datasets, and give
## the IRF model different starting values for each dataset.  Also add some 
## spectral relations between dataset (that are actually equalities, since the
## linear relations are fixed to 1.  The datasets are related as a whole via
## scaling parameters dscal.  For the 3rd dataset dscal is estimated at
## at each wavelength (per-clp).
##############################

didRes<-fitModel(data, list(model_allfree), 
modeldiffs = list(dscal = 
list(
list(to=4,from=1,value=1), 
list(to=2,from=1,value=1), 
list(to=3,from=1,value=rep(1,32), perclp=TRUE)),
change = list(
list(what="kinpar", spec= c(1.01, 0.05, 0.005), dataset=2:4,
type="multifree"),
list(what="irfpar", spec= c(.067, 0.18), dataset=2:4, type="multifree"),

list(what="clpequspec", spec=list(
list(to=3, from=3, low=100, high=1810, startrelpar=1, dataset=3), 
list(to=1, from=1, low=100, high=1810, startrelpar=1, dataset=2)), 
dataset=c(1,4), type="multifree"),

list(what="clpequ", spec=c(1,1), dataset=c(1,4), type="multifree"), 

list(what = "fixed", spec = list(clpequ=c(1,2)), dataset=c(1,4)))),

opt=kinopt(iter=3, linrange=3, superimpose=1:4, 
selectedtraces=seq(1,data[[1]]@nl, by=3)))

##############################
## Add a more sophisticated model for the coherent artifact,
## via the $type = "irfmulti" argument.  The 
## time profile of the IRF is used for the coherent
## artifact/scatter model, but the IRF parameters are taken per
## dataset.
##############################

model_irfcoh <- initModel(mod_type = "kin", 
kinpar=c(1.0, 0.05, 0.005), 
irfpar=c(.05, 0.495912E-01), 
lambdac = 1670,
fixed = list(irfpar=2),
parmu = list(c(-.0158119812957)), 
seqmod=TRUE,iter=2, positivepar=c("kinpar", "coh"), 
weightpar=list( c(-16,.3,1,2000,.1), c(NA,NA,1590,1612,.1)), 
cohspec = list( type = "irfmulti", numdatasets = 4))

##############################
## Fit with this more sophisticated coherent artifact model, 
## and fixing the width of the IRF. 
##############################

didRes<-fitModel(data, list(model_irfcoh), 
modeldiffs = list(dscal = 
list(
list(to=4,from=1,value=1), 
list(to=2,from=1,value=1), 
list(to=3,from=1,value=rep(1,32), perclp=TRUE)),
change = list(
list(what="kinpar", spec= c(1.01, 0.05, 0.005), dataset=2:4,
type="multifree"),
list(what="irfpar", spec= c(.067, 0.18), dataset=2:4, type="multifree"),

list(what="clpequspec", spec=list(
list(to=3, from=3, low=100, high=1810, startrelpar=1, dataset=3), 
list(to=1, from=1, low=100, high=1810, startrelpar=1, dataset=2)), 
dataset=c(1,4), type="multifree"),

list(what="clpequ", spec=c(1,1), dataset=c(1,4), type="multifree"), 

list(what = "fixed", spec = list(clpequ=c(1,2)), dataset=c(1,4)))),
list(what = "fixed", spec = list(clpequ=c(1), irfpar=2), dataset=2:3), 

opt=kinopt(iter=3, linrange=3, superimpose=1:4, 
selectedtraces=seq(1,data[[1]]@nl, by=3)))




