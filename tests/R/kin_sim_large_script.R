require(TIMP) 

## simulate a 501 x 126 matrix of data (501 x 501 is too heavy for CRAN)

times <- seq(50, 350, by=.6)
wavenum <- seq(18000, 28000, by=80)
# wavenum <- seq(18000, 28000, by=20) # for the 501 x 501 simulation matrix of data

E <- matrix(nrow = length(wavenum), ncol = 3)
location <- c(26000, 24000, 20000) 
delta <- c(2000, 3000, 4000)
amp <- c(1, 2, 3)  
E[, 1] <- amp[1] * exp( - log(2) * (2 * (wavenum - location[1])/delta[1])^2)
E[, 2] <- amp[2] * exp( - log(2) * (2 * (wavenum - location[2])/delta[2])^2)
E[, 3] <- amp[3] * exp( - log(2) * (2 * (wavenum - location[3])/delta[3])^2)

PSI <- matrix(nrow=length(times), ncol = length(wavenum))

for (i in 1:length(wavenum)) {
     irfvec <- irfparF(irfpar = c(57.47680283, 1.9), lambdac = 1500, 
     lambda = wavenum[i], i=1, mudisp = TRUE, parmu = c(.001,.001), 
     dispmufun = "poly", taudisp = FALSE, disptaufun="",partau=vector())

     cohirf <- irfparF(irfpar = c(57.47680283, 1.9), lambdac = 1200, lambda =
     wavenum[i], i=1, mudisp = TRUE, parmu = c(.0001,.0001), taudisp = FALSE,
     dispmufun = "poly")

    C <- compModel (k=c(.01,.05), x=times, irfpar =irfvec, cohirf = cohirf, 
    irf = TRUE, cohspec = list(type = "freeirfdisp"),coh = vector(), 
    lamb = i, dataset = 1,usekin2=FALSE) 
    
    PSI[,i] <- C %*% as.matrix(E[i,])    
}    

sigma <- .01
PSI  <- PSI + sigma * rnorm(dim(C)[1] * dim(E)[1])

ser2 <- dat(psi.df = PSI, x = times, nt = length(times), x2 = wavenum, nl =
length(wavenum)) 

model1<- initModel(mod_type = "kin", 
kinpar=c(.01, .05), lambdac = 1200, 
irfpar=c(57.47680283, 1.9), 
parmu = list(c(.001,.001), c(.0001,.0001)), 
seqmod=FALSE, cohspec = list(type="freeirfdisp"),
makeps="Sergey data", title="Ser")

## fit the model 

serRes<-fitModel(list(ser2), list(model1), 
opt=kinopt(iter=1, plot = TRUE))
