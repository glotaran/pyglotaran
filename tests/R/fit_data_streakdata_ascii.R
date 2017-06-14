require(TIMP)
# Determine the local dir:
# http://stackoverflow.com/questions/3452086/getting-path-of-an-r-script
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getSrcDirectory(function(x) {x})
gtaDataset1 <- readData('../resources/data/streakdata.ascii')

# R Call for the TIMP function "initModel":
gtaModel1 <- initModel(mod_type = "kin",
                       kinpar = c(0.15,0.03,0.01,2.0E-4),
                       irffun = "gaus",
                       irfpar = c(-82.0,0.5),
                       streak = TRUE,
                       streakT = 13200.0,
                       positivepar=vector(),
                       cohspec = list(type = ""),
                       fixed = list(clpequ=1:0),
                       seqmod = FALSE)

kmat_seq <- array(0, dim=c(4,4,2))
kmat_seq[2,1,1] <- 1
kmat_seq[3,2,1] <- 2
kmat_seq[4,3,1] <- 3
kmat_seq[4,4,1] <- 4

kmat_par <- array(0, dim=c(4,4,2))
kmat_par[1,1,1] <- 1
kmat_par[2,2,1] <- 2
kmat_par[3,3,1] <- 3
kmat_par[4,4,1] <- 4

jvec_par=c(1,1,1,1)
jvec_seq=c(1,0,0,0)
print("kmat_seq:"); print(kmat_seq)
print("kmat_par:"); print(kmat_par)

gtaModel2 <- initModel(mod_type = "kin",
                       kinpar = c(0.2,0.02,0.07,1.6E-4),
                       irffun = "gaus",
                       irfpar = c(-83,1.5),
                       kmat=kmat_seq,
                       jvec=jvec_seq,
                       streak = TRUE,
                       streakT = 13200.0,
                       fixed = list(jvec=c(1,2,3,4),kinpar=1:4),
                       seqmod = FALSE)

# R Call for the TIMP function "fitModel":
gtaFitResult <- fitModel(data = list(gtaDataset1),
                         modspec = list(gtaModel2),
                         modeldiffs = list(linkclp = list(c(1))),
                         opt = kinopt(iter = 1,
                                      linrange = 20,
                                      nnls = FALSE,
                                      stderrclp = FALSE,
                                      plot = TRUE),
                         lprogress = FALSE)

# Final residual standard error: 44.0835
# Estimated Kinetic parameters: Dataset1: 0.224982, 0.0212250, 0.0680685, 0.000159677
# Standard errors: 0.00147862, 5.21901e-05, 0.000527943, 9.90781e-07
# Estimated Irf parameters: Dataset1: -83.8533, 1.60986
# Standard errors: 0.00296940, 0.00306670



# # R Call for the TIMP function "initModel":
# kmat1 = array(c(0,1,1,1,1,0,0,2,0,0,0,2,4,3,0,0,0,3,0,0,0,0,0,0,5,0,1,2,3,4,0,0,0,0,0,0,5,0,0,0,0,0,6,0,0,0,0,0,0,0), dim = c(5,5,2))
# gtaModel1 <- initModel(mod_type = "kin",
#                        kinpar = c(2.0,0.12687052185780798,0.025221263651435427,0.04,2.0E-4),
#                        kmat = kmat1,
#                        jvec = c(1.0,0.0,0.0,0.0,0.0),
#                        kinscal = c(0.03,0.89,0.03,0.05,0.5,1.3),
#                        irffun = "gaus",irfpar = c(-84.63050138816294,1.5730268178714186), streak = TRUE, streakT = 13200.0,positivepar=vector(),clp0 = list(list(low = 100.0, high =1000.0, comp = 1),list(low = 100.0, high =690.0, comp = 2),list(low = 100.0, high =690.0, comp = 4)),fixed = list(kinpar=c(1), kinscal=c(1,2,3,4), jvec=c(1,2,3,4,5),clpequ=1:0),seqmod = FALSE)
#
# # R Call for the TIMP function "fitModel":
# gtaFitResult <- fitModel(data = list(gtaDataset1),
#                          modspec = list(gtaModel1),
#                          modeldiffs = list(linkclp = list(c(1))),
#                          opt = kinopt(iter = 15,
#                                       nnls = TRUE,
#                                       stderrclp = FALSE,
#                                       plot = TRUE),
#                          lprogress = TRUE)

#Final residual standard error: 44.6350
#
# Estimated Kinetic parameters: Dataset1: 2.00000, 0.128877, 0.0283520, 0.0675294, 0.000163065
# Standard errors: 2.00000, 0.00107970, 0.000361719, 0.000186165, 9.95545e-07
#
# Estimated Irf parameters: Dataset1: -84.3327, 1.54850
# Standard errors: 0.00304614, 0.00324986
#
# Estimated Kinscal: Dataset1: 0.0300000, 0.890000, 0.0300000, 0.0500000, 0.440656, 1.36316
# Standard errors: 0.0300000, 0.890000, 0.0300000, 0.0500000, 0.00565600, 0.0107292
#
# Estimated J vector: Dataset1: 1.00000, 0.00000, 0.00000, 0.00000, 0.00000
# Standard errors: 1.00000, 0.00000, 0.00000, 0.00000, 0.00000