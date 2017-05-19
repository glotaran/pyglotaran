source('_load_TIMP_package_and_set_seed.R')

# load times and times_no_IRF
source('_shared_time_vector.R')
times1 <- times_with_IRF
times2<-times1
# TIMP is not smart enough to have different time length:
# times2 <- c(times_with_IRF, head(seq(3100, 3500, by=100),-1))

ncomp1=2 #first dataset 2 comps
ncomp2=3 #second dataset 3 comps
noise1a = 0.002
noise1b = 0.001

spectral_indices1a <- c(1, 2)
# spectral_indices1b <- c(2, 3) #TIMP cannot handle different c-matrix size for different datasets
spectral_indices1b <- c(3, 4)
spectral_indices2a <- c(10, 20)
spectral_indices2b <- c(30, 40)
spectral_indices3a <- c(680.3, 680.4)
spectral_indices3b <- c(680.31, 680.41)

print(sprintf("Length(wavenum) = %i ", length(wavenum)))
print(sprintf("Number of datapoints = %i ", length(times)*length(wavenum)))
ematrix3a <- ematrix2a <- ematrix1a <- as.matrix(array(seq(1,2,0.5), dim=c(length(spectral_indices1a), ncol = ncomp1)))
ematrix3b <- ematrix2b <- ematrix1b <- as.matrix(array(seq(3,1,-0.5), dim=c(length(spectral_indices1b),ncomp2)))

kmatrixa <- array(0, dim=c(2,2,2))
kmatrixa[2,1,1] <- 1
kmatrixa[2,2,1] <- 2
kmatrixb <- array(0, dim=c(3,3,2))
kmatrixb[2,1,1] <- 1
kmatrixb[2,2,1] <- 2
kmatrixb[3,3,1] <- 3


irf <- TRUE 
cohirf <- FALSE #coherent artifact
# sim_kinpar_params <- c(0.05123, 0.00101, 0.04876) # Too hard for TIMP, 0.04876 and 0.05123 too close together.
sim_kinpar_params <- c(0.05123, 0.00101, 0.014876)
sim_irfpar_params <- c(-0.099999, 0.01)

C1 <- compModel(k=sim_kinpar_params, x=times1, 
                irf = irf, irfpar =sim_irfpar_params, 
                cohirf = cohirf, # cohspec = list(type = "freeirfdisp"),coh = vector(), 
                dataset = 1) 
C2 <- compModel(k=sim_kinpar_params, x=times2, 
                irf = irf, irfpar =sim_irfpar_params, 
                cohirf = cohirf, # cohspec = list(type = "freeirfdisp"),coh = vector(), 
                dataset = 2) 

datamatrix1a <- C1[,1:2] %*% t(ematrix1a) + noise1a * rnorm(dim(C1)[1] * dim(ematrix1a)[1])
datamatrix1b <- C2[,1:3] %*% t(ematrix1b) + noise1b * rnorm(dim(C2)[1] * dim(ematrix1b)[1])

dev.off()
matplot(times1, cbind(datamatrix1a,C1[,1:2] %*% t(ematrix1a)), type="l", lty=1,
        xlab = "time (ps)", ylab = "amp")
matplot(times2, cbind(datamatrix1b,C2[,1:3] %*% t(ematrix1b)), type="l", lty=1,
        xlab = "time (ps)", ylab = "amp")

# Init TIMP dataset objects
dataset1a <- dat(psi.df = datamatrix1a, x = times1, nt = length(times1), x2 = spectral_indices1a, nl = length(spectral_indices1a))
dataset2a <- dat(psi.df = datamatrix1a, x = times1, nt = length(times1), x2 = spectral_indices3a, nl = length(spectral_indices2a)) 
dataset3a <- dat(psi.df = datamatrix1a, x = times1, nt = length(times1), x2 = spectral_indices3a, nl = length(spectral_indices3a)) 
dataset1b <- dat(psi.df = datamatrix1b, x = times2, nt = length(times2), x2 = spectral_indices1b, nl = length(spectral_indices1b))
dataset2b <- dat(psi.df = datamatrix1b, x = times2, nt = length(times2), x2 = spectral_indices2b, nl = length(spectral_indices2b))
dataset3b <- dat(psi.df = datamatrix1b, x = times2, nt = length(times2), x2 = spectral_indices3b, nl = length(spectral_indices3b))

# Init TIMP model
model1a <- initModel(mod_type = "kin",
                         kinpar=0.95*sim_kinpar_params[1:2] , 
                         irfpar=0.95*sim_irfpar_params, 
                         seqmod=FALSE, 
                         makeps="Simulate data", title="sim-large")
model1b <- initModel(mod_type = "kin",
                     kinpar=0.95*sim_kinpar_params, 
                     irfpar=0.95*sim_irfpar_params, 
                     seqmod=FALSE, 
                     makeps="Simulate data", title="sim-large")

## fit the model 
ptm <- proc.time()
fit1a<-fitModel(list(dataset1a), list(model1a), 
                     opt=kinopt(iter=30, plot = TRUE))
time_taken_fitting <- (proc.time() - ptm)
test_fit_noise<-fitModel(list(test_data_noise), list(test_model), 
                   opt=kinopt(iter=30, plot = FALSE))
time_taken_fitting_noise <- (proc.time() - ptm)

## fit the model 

denRes<-fitModel(data = list(dataset1a, dataset1b), list(model1a),
                 modeldiffs = list(thresh = 0.001, dscal = list(list(to=2,from=1,value=1)), 
                                   add = list(
                                     list(what = "kinpar", ind = 3, dataset = 2, start= 0.95*sim_kinpar_params[3])
                                   ),
                                   change = list(list(what = "fixed", dataset=2, spec = list(drel = 1)))),
                 opt=kinopt(iter=3,superimpose = c(1,2), divdrel = TRUE, linrange = 1,
                            xlab = "time (ns)", ylab = "wavelength"
                            #,makeps = "den2"
                            ))



print(sprintf("Fitting time taken: %f", time_taken_fitting[3]))

