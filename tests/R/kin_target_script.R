    
########################################
## Data and model of Delmar Larsen  
######################################## 

##############################
## load TIMP
##############################

require(TIMP)

## read data 
delDat<-readData("PCP1.-50000fs.PP.txt")

## scale wavelength axis 
delDatP<-preProcess(data = delDat, scalx2 = c(1.19, 404),sample_lambda=4)

## set up the K matrix.  

## initialize 2 5x5 arrays to 0 
## replace 5 with the number of columns/rows in the
## desired K matrix 
 
delK <- array(0, dim=c(5,5,2))

## the matrix is indexed: 
## delK[ ROW K MATRIX, COL K MATRIX, matrix number] 

## in the first matrix, put the label of compartments 
## that are non-zero 

delK[2,1,1] <- 1
delK[5,1,1] <- 1
delK[3,2,1] <- 2
delK[5,2,1] <- 2
delK[4,3,1] <- 3
delK[5,3,1] <- 3
delK[5,4,1] <- 4
delK[5,5,1] <- 5


## in the second matrix, put the indices of any branching
## parameters you want; these are the parameter values given
## in the vector kinscal and then fixed  

delK[2,1,2] <- 1
delK[5,1,2] <- 3
delK[2,2,2] <- 2
delK[3,2,2] <- 4
delK[5,2,2] <- 5
delK[3,3,2] <- 2
delK[4,3,2] <- 5
delK[5,3,2] <- 7
delK[4,4,2] <- 2

## print out the resulting array to make sure it's right

delK

##, , 1
##
##     [,1] [,2] [,3] [,4] [,5]
## [1,]    0    0    0    0    0
## [2,]    1    0    0    0    0
## [3,]    0    2    0    0    0
## [4,]    0    0    3    0    0
## [5,]    1    2    3    4    5
##
## , , 2
##
##     [,1] [,2] [,3] [,4] [,5]
## [1,]    0    0    0    0    0
## [2,]    1    2    0    0    0
## [3,]    0    4    2    0    0
## [4,]    0    0    5    2    0
## [5,]    3    5    7    0    0

## in setting up the model, put the branching parameters
## in kinscal and then fix their values. 

delMod<-initModel(mod_type="kin",
kinpar=c(25,5,.9731,.2489,.013), 
kinscal=c(.73,.05,.3,.75,.24,.47,.4), 
irfpar=c(-.096, .022), kmat=delK, jvec=c(1,0,0,0,0),
seqmod = FALSE,
lambdac=500,
positivepar=c("kinpar"),
weightpar=list(c(NA,NA,460,525,.3)),
clp0=list(list(comp=5, low=250,high=550), list(comp=5, 
low=600,high=800)), 
fixed=list(kinpar=c(1,2),kinscal=c(1:7), jvector=1:5))

serRes<-fitModel(list(delDatP), list(delMod), 
opt=kinopt(iter=1, linrange = 20,
makeps = "ser", plotkinspec = TRUE,
selectedtraces = seq(1, delDatP@nl, by=20),
xlab = "time (ps)", ylab = "wavelength"))
