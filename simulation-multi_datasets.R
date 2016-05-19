## If the R package TIMP is not installed, please run:
## install.packages("TIMP")

require(TIMP) 
set.seed(123)

## Define kmatrix
m1K <- array(0, dim=c(3,3,2)) 
m1K[1,1,1] <- 5
m1K[2,1:3,1] <- c(1,4,3)
m1K[3,2:3,1] <- c(2,5)
rates = c(15, 8, 0.667, 20, 0.2456789)
inputs = c(0.45, 0.45, 0.1)

print(sprintf("User specified k-matrix:") )
write.table(format(m1K[,,1], justify="right"),
            row.names=F, col.names=F, quote=F)

# Verify the construction of the real K-matrix
fillK(kmat=m1K,theta=rates,fixedkmat=FALSE,kinscalspecialspec=c())
# Verify the calculation of the A matrix and the eigenvalues
fullKF(kmat=m1K,theta=rates,jvec=inputs)

# generate spectral shapes
wavenum <- seq(13334.0, 15384.00, by=41.0)
location <- c(14814.815, 14641.288, 14044.944) 
delta <- c(400, 600, 500)
amp <- c(1, 1, 1)

E <- matrix(nrow = length(wavenum), ncol = length(location))

for (i in 1:length(location)) {
  E[, i] <- amp[i] * exp( - log(2) * (2 * (wavenum - location[i])/delta[i])^2)
}

# generate timepoints
times1 <- seq(-0.001, 0.04950, by=0.0005)
times2 <- seq(0.0500, 2.00000, by=0.0200)
times <- c(times1,times2)

# define IRF
irf = TRUE
irfvec <- c(0.0034,0.0002) #location width
dispvec <- c(-.0001,.0001)

psisim1a <- matrix(nrow=length(times), ncol = length(wavenum))
psisim1b <- matrix(nrow=length(times), ncol = length(wavenum))

C <- compModel (k=rates, x=times, irfpar =irfvec, jvec=inputs, irf = irf, fullk = TRUE, kmat=m1K) 

for (i in 1:length(wavenum)) {
  psisim1a[,i] <- C %*% as.matrix(E[i,])    
}    

for (i in 1:length(wavenum)) {
  ## This is code for generating dispersion (per wavelength dependent irf location)
       tempirfvec <- irfparF(irfpar = irfvec, lambdac = 14334, 
       lambda = wavenum[i], i=1, #i is only used when dispmufun="discrete"
       mudisp = TRUE, dispmufun = "poly", parmu = dispvec, 
       taudisp = FALSE, disptaufun="",partau=vector())
       
       tempC <- compModel (k=rates, x=times, irfpar =tempirfvec, jvec=inputs, irf = irf, fullk = TRUE, kmat=m1K) 
  
       psisim1b[,i] <- tempC %*% as.matrix(E[i,])    
}  

# Add noise to the data (optionally) and plot:
sigma <- .000
psisim1a  <- psisim1a + sigma * rnorm(dim(C)[1] * dim(E)[1])
contour(times,wavenum,psisim1a,levels=seq(0,1,0.02))

## FITTING

dat1a <- dat(psi.df = psisim1a, x = times, nt = length(times), x2 = wavenum, nl = length(wavenum)) 

start1a <- -fullKF(kmat=m1K,theta=rates,jvec=inputs)$values

model1a<- initModel(mod_type = "kin", 
                   kinpar=0.9*start1a, 
                   irfpar=1.1*irfvec, 
                   irf = TRUE,
                   seqmod=TRUE, 
                   makeps="Simulated data - no noise", title="data1")

## fit the model1a to dat1a 
ptm <- proc.time()
res1a<-fitModel(list(dat1a), list(model1a), opt=kinopt(iter=30, xlab = "time (ns)", 
                 ylab = "wavenumbers (/cm)", selectedtraces = seq(6,50,by=5), 
                 algorithm = "nls",
                 plot = TRUE))
print(sprintf("Model 1a: time taken: %f", (proc.time() - ptm)[3]))


sigma <- .002
psisim1b  <- psisim1b + sigma * rnorm(dim(C)[1] * dim(E)[1])
contour(times,wavenum,psisim1b,levels=seq(0,1,0.02))

dat1b <- dat(psi.df = psisim1b, x = times, nt = length(times), x2 = wavenum, nl = length(wavenum)) 

start1b <- -fullKF(kmat=m1K,theta=rates,jvec=inputs)$values

## fit the model1b to dat1b
start1b <- -fullKF(kmat=m1K,theta=rates,jvec=inputs)$values
model1b<- initModel(mod_type = "kin", 
                   kinpar=0.99*start1b, # lambdac = 1200, 
                   irfpar=1.01*irfvec, 
                   irf = TRUE,
                   parmu = list(1.05*dispvec), 
                   seqmod=TRUE, 
                   lambdac = 14334,
                   makeps="Simulated data - noise", title="data1")

ptm <- proc.time()
res1b<-fitModel(list(dat1b), list(model1b), opt=kinopt(iter=30, xlab = "time (ns)", 
                                                      ylab = "wavenumbers (/cm)", selectedtraces = seq(6,50,by=5), 
                                                      algorithm = "nls",
                                                      plot = TRUE))
print(sprintf("Model 1b: time taken: %f", (proc.time() - ptm)[3]))
