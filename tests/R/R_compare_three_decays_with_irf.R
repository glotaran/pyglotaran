## If the R package TIMP is not installed, please run:
## install.packages("TIMP")

require(TIMP) 
set.seed(123)
sigma <- .000001

irf = TRUE
sim_irfvec_params <- c(0.04,0.5) #location, width
initial_irf_params <- c(-.05,0.1) 
initial_irf_params <- c(0.28,0.28)
# irffun = "gaus" #default is gaus

sim_kinpar_params <- c(101e-3, 202e-4, 505e-5)
initial_kinpar_params <- c(123e-3, 234e-4, 567e-5) 

times <- c(head(seq(-10, -1, by=0.1),-1),
           head(seq(-1, 10, by=0.01),-1),
           head(seq(10, 50, by=1.5),-1),
           head(seq(50, 1000, by=15),-1))
print(sprintf("Length(times) = %i ", length(times)))
## simulate a 1022 x 501 matrix of data
wavenum <- seq(12820, 15120, by=4.6)
print(sprintf("Length(wavenum) = %i ", length(wavenum)))
print(sprintf("Number of datapoints = %i ", length(times)*length(wavenum)))

#             680nm  740nm  705nm
location <- c(14700, 13515, 14180) 
delta <- c(400, 100, 300)
amp <- c(1, 1, 1)
E <- matrix(nrow = length(wavenum), ncol = length(location))
for (i in 1:length(location)) {
E[, i] <- amp[i] * exp( - log(2) * (2 * (wavenum - location[i])/delta[i])^2)
}
psisim <- matrix(nrow=length(times), ncol = length(wavenum))

C <- compModel (k=sim_kinpar_params, x=times, irfpar =sim_irfvec_params, # cohirf = cohirf, 
irf = irf, # cohspec = list(type = "freeirfdisp"),coh = vector(), 
lamb = i, dataset = 1,usekin2=FALSE) 
       
for (i in 1:length(wavenum)) {
    psisim[,i] <- C %*% as.matrix(E[i,])    
}    

psisim  <- psisim + sigma * rnorm(dim(C)[1] * dim(E)[1])

data3decays <- dat(psi.df = psisim, x = times, nt = length(times), x2 = wavenum, nl = length(wavenum)) 

model3decays<- initModel(mod_type = "kin", 
kinpar=initial_kinpar_params, # lambdac = 1200, 
irfpar=initial_irf_params, 
irf = TRUE,
seqmod=FALSE, 
makeps="Simulate data", title="sim-large")

## fit the model 
ptm <- proc.time()
fit3decays<-fitModel(list(data3decays), list(model3decays), 
opt=kinopt(iter=30, plot = FALSE))
print(sprintf("Fitting time taken: %f", (proc.time() - ptm)[3]))

examineFit(fit3decays,opt=kinopt(iter=30, xlab = "time (ps)", 
           ylab = "wavelength", selectedtraces = seq(1,256,by=20), plot = TRUE))

#your function here
print(sprintf("Total time taken (including plots): %f", (proc.time() - ptm)[3]))




