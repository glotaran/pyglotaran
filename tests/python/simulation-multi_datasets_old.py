import numpy as np
import scipy.optimize
import scipy.linalg.lapack as lapack
import matplotlib.pyplot as plt
import time

from calculateC import calculateC, calculateCirf


def qr(a, c):
    qr, tau, _, _ = lapack.dgeqrf(a, overwrite_a=1)
    c, _, _ = lapack.dormqr("L", "T", qr, tau, c, max(1, a.shape[1]), overwrite_c=1)
    for i in range(a.shape[1]):
        c[i] = 0
    c, _, _ = lapack.dormqr("L", "N", qr, tau, c, max(1, a.shape[1]), overwrite_c=1)
    return c

def irfparF(par_0, par_disp=np.empty((0,)), l=0, l_c=0):
    disp_s = par_0
    for i in range(par_disp.size):
        disp_s += par_disp[i] * np.power(((l - l_c) / 100), i + 1)
    return disp_s


def fillK(kmat):
    idx = np.diag_indices(kmat.shape[0])
    kmat[idx] = - kmat.sum(axis=0)
    return kmat


def fullKF(kmat, jvec):
   kshape = kmat.shape[0]
   A = np.empty((kshape, kshape), dtype=np.float64)
   K = fillK(kmat)
   eigenValues, eigenVectors = np.linalg.eig(K)
   idxsort = eigenValues.argsort()[::1]   
   eigenValues = eigenValues[idxsort]
   eigenVectors = eigenVectors[:,idxsort]
   print(eigenValues, eigenVectors)
   gamma = np.matmul(scipy.linalg.inv(eigenVectors), jvec)
   for i in range(kshape):
       A[i,:] = eigenVectors[:,i] * gamma[i]
   return(A, eigenValues)

def compModel(kinpar, times, wavenum, irf=False, mu=0.0, delta=0.0,\
                fullk=False, kmat=np.empty(0,), jvec=np.empty(0,)):
    if fullk:
       eig = fullKF(kmat, jvec)
       k = - eig[1]
       A = eig[0]
    else:
       k = kinpar

    C = np.empty((times.shape[0], k.shape[0]), dtype=np.float64)    
    
    if irf:
        calculateCirf(C, k, times, mu, delta)
    else:
        calculateC(C, k, times)
        
    if fullk:
        C = np.matmul(C, A)
    
    return C

def splitVector(params, m_k1, n_k1, m_k2, n_k2, n_j1, n_j2):
    mu_0 = np.nan
    delta_0 = np.nan
    kmat1 = np.empty((0,))
    kmat2 = np.empty((0,))
    jvec1 = np.empty((0,))
    jvec2 = np.empty((0,))
    idx = 0
    kmat1 = params[idx: idx + m_k1 * n_k1]
    idx += m_k1 * n_k1
    kmat2 = params[idx: idx + m_k2 * n_k2]
    idx += m_k2 * n_k2
    jvec1 = params[idx: idx + n_j1]
    idx += n_j1
    jvec2 = params[idx: idx + n_j2]
    idx += n_j2
    mu_0 = params[idx]
    delta_0 = params[idx + 1]
    return (kmat1.reshape(m_k1, n_k1), kmat2.reshape(m_k2, n_k2), jvec1, jvec2, mu_0, delta_0)
    
    

def solve(params, PSI1, PSI2, times, wavenum, m_k1, n_k1, m_k2, n_k2, n_j1, n_j2):
    res1 = np.empty(PSI1.shape, dtype=np.float64)
    res2 = np.empty(PSI2.shape, dtype=np.float64)
    
    (kmat1, kmat2, jvec1, jvec2, mu_0, delta_0)=\
        splitVector(params, m_k1, n_k1, m_k2, n_k2, n_j1, n_j2)

    for i in range(PSI1.shape[1]):
        mu = mu_0
        delta = delta_0
        C = compModel(None, times, wavenum, True, mu, delta, True, kmat1, jvec1) #Needs to be expanded
        
        b = PSI1[:,i]
        res1[:,i] = qr(C, b)
        #res[:,i] = scipy.optimize.nnls(C, b)[1]
        
    for i in range(PSI2.shape[1]):
        mu = mu_0
        delta = delta_0
        C = compModel(None, times, wavenum, True, mu, delta, True, kmat2, jvec2) #Needs to be expanded
        
        b = PSI2[:,i]
        res2[:,i] = qr(C, b)
        #res[:,i] = scipy.optimize.nnls(C, b)[1]

    return np.hstack((res1.flatten(), res2.flatten()))


def main():
    # The following code comments are to be seen as a rough draft of the 
    # simulation code that is to be implemented below,
    # and can be removed when the test case is written.
    #    
    # In this simulation 3 datasets will be simulated
    #   in the spectral window from 650nm to 750nm
    #   equidistant in wavenumbers
    # Dataset 1 will be 51 wavelengths * 201 timepoints
    #   From 650 nm (15384/cm) to 750 nm (13334/cm)
    #   Spectral stepsize: (15384-13334)/50 = 41
    #       15384, 15343, ... , 13334
    #   From -0.5 ps to 2ns
    #       From -1 ps to 50 ps in steps of 0.5 ps (103 timepoints)
    #       From 50 ps to 2000 ps in steps of 20 ps (98 timepoints)
    #       or linearly spaced on a logarithmic axis, e.g.:
    #       np.exp(np.log(50)+i*(np.log(2000)-np.log(50))/99) for i=1:99
    # Dataset 2 will be 41 wavelengths * 251 timepoints 
    #   From 666 nm (15015/cm) to 750 nm (13334/cm)
    #   Spectral stepsize: (15015-13334)/41 = 41
    #       15015, 15056, ... , 13334
    # Dataset 3 will be 51 * 2-3 SAS guidance spectra for megacomplex 1
    # Dataset 4 will be 51 * 1 Steady State Spectra for <to be defined> 
    # The dynamics in the datasets is characterized by the linear sum of the
    # contributions from : 2 megacomplexes 
    # Megacomplex 1 contains 3 compartments
    # C1` -> C2^ <-> C3 
    # with intrinsic decay (`) from C1 and C3 and trapping from C2
    # C1: represents blue antenna
    # C2: represnets bulk from which trapping takes place
    # C3: represents a red trap from which can (only) detrap via C2
    # Megacomplex 2 contains 4 compartments 
    # C4` <-> C5` <-> C6^ 
    # irreversible trapping from C6 (^) 
    # C4 reprsents antenna in equilibrium with bulk C5
    # C6 represents a dark (no spectrum) CT state from which either
    # irreversible trapping can take place or detrapping to the bulk C5
    # intrisic decay occurs from C4 and C5   
    # define properties megacomplexes
    # megacomplex 1 (m1)
    # kmatrix1 before fillK
    #   from   C1        C2         C3
    # to C1    k5         0          0  
    # to C2    k1        k4         k3      
    # to C3    0         k2         k5
    # kmatrix after fillK (for unit test)
    #   from   C1        C2         C3
    # to C1   -k5-k1      0          0  
    # to C2    k1       -k4-k2      k3      
    # to C3    0         k2         -k5-k3   
    # megacomplex 2 (m2)
    # kmatrix2 ebfore fillK (work in progress)
    #   from   C4        C5         C6     C7
    # to C4    k7        k2          0      0
    # to C5    k1        k7         k6      k4
    # to C6     0        k5         k7      0
    # to C7     0        k3          0      k8
    # kmatrix after fillK
    #   from   C4        C5         C6
    # to C4  -k1-k6   k2-k3-k6   -(k4+k5)  
    # to C5    k1        k6         k4
    # to C6     0        k3         k5  
    
    # Define k-matrix for m1    
    (k1, k2, k3, k4, k5) = (15, 8, 0.667, 20, 0.2456789) # rates in /ns (per nanosecond)    
    jvec1 = np.asarray([0.45, 0.45, 0.1])
    kmatrix1 = np.asarray([[k5, 0, 0], [k1, k4, k3], [0, k2, k5]])    
    (test_rates, test_A) = fullKF(kmatrix1, jvec1)# call fullKF
    
    # Generate spectral shapes for m1
    wavenum = np.asarray(np.arange(13334.0, 15384.00, 41.0))        
    location1 = np.asarray([14814.815, 14641.288, 14044.944])
    delta1 = np.asarray([400,600,500])
    amp1 = np.asarray([1, 1, 1])        
    E1 = np.empty((wavenum.size, location1.size), dtype=np.float64)
    for i in range(location1.size):
        E1[:, i] = amp1[i] * np.exp(-np.log(2) * np.square(2 * (wavenum - location1[i]) / delta1[i]))    
    
     #    

    # get k and A
    
    #print(kmatrix1)
    
    
    location2 = np.asarray([14727.541, 14598.540, 14388.490, 14388.490])
    delta2 = np.asarray([350,600,400, 100])
    amp2 = np.asarray([1, 1, 1, 0])

    (k1, k2, k3, k4, k5, k6, k7, k8) = (30, 10, 4.2, 1.6, 5.5, 2.4, 0.2456789, 3.3)
    jvec2 = np.asarray([0.6, 0.3, 0.05, 0])
    kmatrix2 = np.asarray([[k7, k2, 0, 0], [k1, k7, k6, k4], [0, k5, k7, 0], [0, k3, 0, k8]])
    # call fillK
    (k2, A2) = fullKF(kmatrix2, jvec2)# call fullKF
    # get k and A
    
    #Generate dataset 1    
    times1 = np.asarray(np.arange(-0.001, 0.05001, 0.0005))
    times2 = np.asarray(np.arange(0.0500, 2.0000, 0.0200))
    times = np.hstack((times1, times2))
    
    (mu_0, delta_0) = (0.0034, 0.0002)
    

    # take 0.2 m1 and 0.7 m2
    
    #Generate dataset 2    
    times1 = np.asarray(np.arange(-0.001, 0.05001, 0.0005))
    times2 = np.asarray(np.arange(0.0500, 3.00000, 0.0200))
    times = np.hstack((times1, times2))
    wavenum = np.asarray(np.arange(13334.0, 15015, 41.0))
    E2 = np.empty((wavenum.size, location2.size), dtype=np.float64)
    
    for i in range(location2.size):
        E2[:, i] = amp2[i] * np.exp(-np.log(2) * np.square(2 * (wavenum - location2[i]) / delta2[i]))

    PSI1 = np.empty((times.size, wavenum.size), dtype=np.float64)
    for i in range(wavenum.size):
        mu = mu_0
        delta = delta_0
        C1 = compModel(None, times, wavenum, True, mu, delta, True, kmatrix1, jvec1)
        C2 = compModel(None, times, wavenum, True, mu, delta, True, kmatrix2, jvec2)
        PSI1[:,i] = np.matmul(0.3 * C1, E1[i,:]) + np.matmul(0.7 * C2, E2[i,:])
    
    PSI2 = np.empty((times.size, wavenum.size), dtype=np.float64)
    for i in range(wavenum.size):
        mu = mu_0
        delta = delta_0
        C1 = compModel(None, times, wavenum, True, mu, delta, True, kmatrix1, jvec1)
        C2 = compModel(None, times, wavenum, True, mu, delta, True, kmatrix2, jvec2)
        PSI2[:,i] = np.matmul(0.6 * C1, E1[i,:]) + np.matmul(0.4 * C2, E2[i,:])
    
    # take 0.6 m1 and 0.4 m2
    start_kmat1 = kmatrix1 + 0.03 * np.random.randn(kmatrix1.shape[0], kmatrix1.shape[1])
    start_kmat2 = kmatrix2 + 0.03 * np.random.randn(kmatrix2.shape[0], kmatrix2.shape[1])
    start_jvec1 = jvec1 + 0.03 * np.random.randn(jvec1.shape[0])
    start_jvec2 = jvec2 + 0.03 * np.random.randn(jvec2.shape[0])
    start_mu = 0.0038
    start_delta = 0.00025
    params = np.hstack((start_kmat1.flatten(), start_kmat2.flatten(), start_jvec1, start_jvec2, start_mu, start_delta))
    ## the rest of the code still needs to be updated
    
    res = scipy.optimize.leastsq(solve, params, args=(PSI1, PSI2, times, wavenum,\
        start_kmat1.shape[0], start_kmat1.shape[1], start_kmat2.shape[0],\
        start_kmat2.shape[1], start_jvec1.shape[0], start_jvec2.shape[0]), full_output=0)
        
    print(res)
    
if __name__ == '__main__':
    main()
