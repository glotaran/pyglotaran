##############################
## Datasets of Rob Koehorst, Bart van Oort, 
## Sergey Laptenok, Ton Visser and Herbert van Amerongen
## 
##
## The below commands repeat the case study in spectral 
## modeling in the paper  "TIMP: an R package for 
## modeling multi-way spectroscopic measurements", 
## Mullen and van Stokkum, submitted to the 
## Journal of Statistical Software. 
##
## Note that the examples require the package TIMP and 
## use of R version 2.5 or higher. 
##############################

require(TIMP)
psi_1 <- readData("psitspec.txt")

psi_1_full <-preProcess(psi_1, sel_time=c(178, 478), 
sel_lambda_ab = c(440, 640))

psi_1_sampled <- preProcess(psi_1, sel_time=c(178, 478), 
sel_lambda_ab = c(440, 640), sample_time=5)

##############################
## linear time dependence of spectral parameters
##############################

model_polylin <- initModel(mod_type = "spec", 
specpar=list(c(20000,3100,-.3) ), specdisp = TRUE,
specdispindex = list(c(1,1), c(1,2), c(1,3)),
specdisppar=list(c(-2000),c(1), c(.2)),
specref=53, specfun="gaus")

res_polylin <- fitModel(data = list(psi_1_sampled), 
modspec = list(model_polylin), 
opt=specopt(iter=7, linrange = 20, stderrclp = TRUE, 
plotkinspec = TRUE, kinspecerr = TRUE, 
makeps = "polylin",
title = "Polynomial parameterization of time dep.",
selectedspectra = seq(1, psi_1_sampled@nt, by=7),
residplot = TRUE, 
xlab = "time", ylab = "wavelength"))

##############################
## Linked exp. rates
##############################

model_exp_linkedrates <- initModel(mod_type = "spec", 
specpar=list(c(18000, 3200, -.1)), specdisp = TRUE,
specdispindex = list(c(1,1), c(1,2), c(1,3)),
specdisppar=list(c(600,1/20), c(400), c(.1)),
specref=53, specfun="gaus",  parmufunc = "exp")

res_model_exp_linkedrates <- fitModel(data = list(psi_1_sampled), 
modspec = list(model_exp_linkedrates), 
opt=specopt(iter=5, linrange = 20, residplot=TRUE,
makeps = "explinked", stderrclp = TRUE,
title = "Exponential parameterization of time dep., linked rates",
plotkinspec = TRUE, kinspecerr = TRUE, superimpose = 1,
selectedspectra = seq(1, psi_1_sampled@nt, by=7),
xlab = "time", ylab = "wavelength")) 
