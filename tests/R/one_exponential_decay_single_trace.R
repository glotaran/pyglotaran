source('_load_TIMP_package_and_set_seed.R')

# load times and times_no_IRF
source('_shared_time_vector.R')
times <- times_no_IRF
psisim <- matrix(nrow=length(times), ncol = 1)
wavenum <- c(685.0)

print(sprintf("Length(wavenum) = %i ", length(wavenum)))
print(sprintf("Number of datapoints = %i ", length(times)*length(wavenum)))

E <- matrix(nrow = length(wavenum), ncol = 1)
E[1,1] <- 1

irf <- FALSE 
cohirf <- FALSE
sim_kinpar_params <- c(0.01)
init_fit_kinpar_params <- c(0.03)

C <- compModel(k=sim_kinpar_params, x=times, 
                irf = irf, #irfpar =sim_irfvec_params, 
                cohirf = cohirf, # cohspec = list(type = "freeirfdisp"),coh = vector(), 
                lamb = wavenum, 
                dataset = 1) 

psisim[,1] <- C %*% as.matrix(E[1,])
psisim_noise  <- psisim + sigma * rnorm(dim(C)[1] * dim(E)[1])
matplot(times, cbind(psisim,psisim_noise), type="l", lty=1,
        xlab = "time (ps)", ylab = "amp")

# Init TIMP dataset objects
test_data <- dat(psi.df = psisim, x = times, nt = length(times), x2 = wavenum, nl = length(wavenum)) 
test_data_noise <- dat(psi.df = psisim_noise, x = times, nt = length(times), x2 = wavenum, nl = length(wavenum)) 

# Init TIMP model
test_model <- initModel(mod_type = "kin", 
                         kinpar=init_fit_kinpar_params, 
                         seqmod=FALSE, 
                         makeps="Simulate data", title="sim-large")

## fit the model 
ptm <- proc.time()
test_fit<-fitModel(list(test_data), list(test_model), 
                     opt=kinopt(iter=30, plot = FALSE))
time_taken_fitting <- (proc.time() - ptm)
test_fit_noise<-fitModel(list(test_data_noise), list(test_model), 
                   opt=kinopt(iter=30, plot = FALSE))
time_taken_fitting_noise <- (proc.time() - ptm)


examineFit(test_fit_noise,opt=kinopt(iter=30, xlab = "time (ps)", 
                                 ylab = "wavelength", selectedtraces = 1, plot = TRUE))


print(sprintf("Fitting time taken: %f", time_taken_fitting[3]))
print(sprintf("Fitting time taken (with noise): %f", time_taken_fitting_noise[3]))
print(sprintf("Total time taken (including plots): %f", (proc.time() - ptm)[3]))
