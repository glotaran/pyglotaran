## If the R package TIMP is not installed, please run:
## install.packages("TIMP")

require(TIMP) 
require(paramGUI)
set.seed(123)

rates = c(0.0101,.00202)
amp = c(1,2)

# generate timepoints
times <- seq(0, 1500, by=1.5)
# times1 <- seq(-0.001, 0.04950, by=0.0005)
# times2 <- seq(0.0500, 2.00000, by=0.0200)
# times <- c(times1,times2)

# define IRF
irf = FALSE
# irfvec <- c(0.0034,0.0002) #location width
# dispvec <- c(-.0001,.0001)

psi <- matrix(nrow=length(times), ncol = 1)

C <- compModel (k=rates, x=times,  irf = irf) 

psi <- C %*% as.matrix(amp)    
   
# Add noise to the data (optionally) and plot:
sigma <- .01000
psin  <- psi + sigma * rnorm(length(times))
#matplot(times,cbind(psin,psi),type="l")

## FITTING

data <- dat(psi.df = psin, x = times, nt = length(times))

modeln<- initModel(mod_type = "kin", 
                   kinpar=0.1*rates, 
                   #irfpar=1.1*irfvec, 
                   #irf = TRUE,
                   seqmod=FALSE, 
                   makeps="Simulated data - with noise", 
                   title="datan")
                                      

## fit the model to the data
ptm <- proc.time()
result <-fitModel(list(data), list(modeln), opt=kinopt(iter=30, xlab = "time (ns)", 
                 algorithm = "nls",
                 plot = FALSE))
                 
                 
nt <- data@nt
nl <- data@nl
x <- data@x
x2 <- data@x2
observed <- data@psi.df
svdobserved <- svd(observed) 
xnew <- x
lin <- max(data@x)
theta <- result$currTheta[[1]]
X <- result$currModel@fit@resultlist[[1]]@cp[[1]]

op <- par(no.readonly = TRUE)
                 
par(mfrow = c(2, 2), oma = c(0, 0, 3, 0))
residuals <- result$currModel@fit@resultlist[[1]]@resid[[1]]

plot(x = x, y = observed, xlab = "time (ps)", ylab = "",
      main = "Data", type = "l", xlim = c(min(x), max(x)))
    
    lines(x = x, y = observed - residuals, col = "red")
    
    abline(0, 0)

fittedC <- compModel(k = theta@kinpar, x = x)     

matplot(x, C, xlab = "time (ps)", ylab = "",
          main = "Concentrations", type = "l", lty = 1)
          
barplot(X, main = "Amplitudes", ylab = "", xlab = "component",
        lty = 1)
          
linlogplot(x = x, y = residuals, mu = 0,
          alpha = lin, xlab = "time (ps)", ylab = "",
          main = "Residuals", type = "l", xlim = c(min(x),
          max(x)))
        abline(0, 0)          
        
kinest <- paste("Kin par:", toString(signif(theta@kinpar, digits = 4)))
mtext(kinest, side = 3, outer = TRUE, line = 1)
