## Example in modeling second order kinetics, by
## David Nicolaides.

## On simulated data.

##############################
## load TIMP
##############################

library("TIMP")

##############################
## SIMULATE DATA
##############################

## set up the Example problem, a la in-situ UV-Vis spectroscopy of a simple
## reaction.
## A + 2B -> C + D, 2C -> E

cstart <- c(A = 1.0, B = 0.8, C = 0.0, D = 0.0, E = 0.0)
times <- c(seq(0,2, length=21), seq(3,10, length=8))
k <- c(kA = 0.5, k2C = 1)

## stoichiometry matrices as per
## Puxty, G., Maeder, M., and Hungerbuhler, K. (2006) Tutorial on the fitting
## of kinetics models to mulivariate spectroscopic measurements with
## non-linear least-squares regression, Chemometrics and Intelligent
## Laboratory Systems 81, 149-164.

rsmatrix <- c(1,2,0,0,0,0,0,2,0,0)
smatrix <- c(-1,-2,1,1,0,0,0,-2,0,1)
concentrations <- calcD(k, times, cstart, rsmatrix, smatrix)

wavelengths <- seq(500, 700, by=2)
spectra <- matrix(nrow = length(wavelengths), ncol = length(cstart))
location <- c(550, 575, 625, 650, 675)
delta <- c(10, 10, 10, 10, 10)
spectra[, 1] <- exp( - log(2) * (2 * (wavelengths - location[1])/delta[1])^2)
spectra[, 2] <- exp( - log(2) * (2 * (wavelengths - location[2])/delta[2])^2)
spectra[, 3] <- exp( - log(2) * (2 * (wavelengths - location[3])/delta[3])^2)
spectra[, 4] <- exp( - log(2) * (2 * (wavelengths - location[4])/delta[4])^2)
spectra[, 5] <- exp( - log(2) * (2 * (wavelengths - location[5])/delta[5])^2)

sigma <- .001
Psi_q <- concentrations %*% t(spectra) + sigma *
  rnorm(dim(concentrations)[1] * dim(spectra)[1])

## store the simulated data in an object of class "dat"
kinetic_data <- dat(psi.df=Psi_q , x = times, nt = length(times),
 x2 = wavelengths, nl = length(wavelengths))

##############################
## DEFINE MODEL 
##############################

## starting values
kstart <- c(kA = 1, k2C = 0.5)

## model definition for 2nd order kinetics
kinetic_model <- initModel(mod_type = "kin", seqmod = FALSE, kinpar = kstart,
                           numericalintegration = TRUE,
                           initialvals = cstart, 
                           reactantstoichiometrymatrix = rsmatrix, 
                           stoichiometrymatrix = smatrix )

##############################
## FIT INITIAL MODEL 
## adding constraints to non-negativity of the
## spectra via the opt option nnls=TRUE
##############################

kinetic_fit <- fitModel(data=list(kinetic_data), modspec = list(kinetic_model),
                        opt = kinopt(nnls = TRUE, iter=80,
                          selectedtraces = seq(1,kinetic_data@nl,by=2)))

## look at estimated parameters

parEst(kinetic_fit)

## make a png of various results

## concentrations 

conRes <- getX(kinetic_fit)

matplot(times, conRes, type="b", col=1,pch=21, bg=1:5, xlab="time (sec)",
        ylab="concentrations", main="Concentrations (2nd order kinetics)")
                        
                        
## spectra 

specRes <- getCLP(kinetic_fit)

matplot(wavelengths, specRes, type="b", col=1,pch=21, bg=1:5,
        xlab="wavelength (nm)",
        ylab="amplitude", main="Spectra")


## see help(getResults) for how to get more results information from
## kinetic_fit
