require(TIMP)

simulateAndExportDatasetTIMP <- function(filename, ...) {
  dataset <- TIMP::simndecay_gen(...)
  cat("filename\n",file=filename)
  cat("dataset name\n",file=filename,append=TRUE)
  cat("Wavelength explicit\n",file=filename,append=TRUE)
  cat("Intervalnr ",  length(dataset@x2),"\n",file=filename,append=TRUE)
  write.table(
    x = dataset@psi.df,
    col.names = dataset@x2,
    row.names = dataset@x,
    sep = " ",
    quote = FALSE,
    file=filename,append=TRUE
  )
}

simulateAndExportDatasetTIMP("simData1.ascii",
                             kinpar = c(0.055, 0.005) , 
                             amplitudes =  c(1, 1) , 
                             tmax =  80 , deltat=  1 , 
                             specpar=  list(c(22000, 4000, 0.1), c(20000, 3500, -0.1)) , 
                             lmin=  400 , lmax=  600 , deltal=  5 , sigma=  0.05 , 
                             irf =  FALSE , irfpar = c( 2 , 1 ) , 
                             seqmod = FALSE )

simulateAndExportDatasetTIMP("simData2.ascii",
                             kinpar = c(0.055, 0.005), 
                             amplitudes =  c(1, 1) , 
                             tmax =  80 , deltat=  1 , 
                             specpar=  list(c(22000, 4000, 0.1), c(20000, 3500, -0.1)) , 
                             lmin=  400 , lmax=  600 , 
                             deltal=  5 , sigma=  0.05 , 
                             irf =  FALSE , irfpar = c( 2 , 1 ) , 
                             seqmod = FALSE )

test_simData1<-readData("simData1.ascii")
test_simData2<-readData("simData2.ascii")