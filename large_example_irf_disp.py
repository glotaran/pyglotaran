import numpy as np
import scipy.optimize
import scipy.linalg.lapack as lapack
import time
import scipy.linalg

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


def fillK(theta, kinscal, kmat, fixedkmat, kinscalspecial, \
            kinscalspecialspec, nocolsums=False):
    kshape = kmat.shape[0]
    reskmat = np.empty(kshape, dtype=np.float64)
    if not fixedkmat:
        special = kinscalspecialspec.shape[0] > 0
        for i in range(reskmat.shape[0]):
            for j in range(reskmat.shape[1]):
                if not (special or nocolsums):
                    if i == j:
                        pl = -1
                    else:
                        pl = 1
                else:
                    pl = 1
                if kmat[i,j,1] != 0 and kmat[i,j,2] != 0:
                    reskmat[i,j] = pl * theta[kmat[i, j, 1]] *\
                    kinscal[kmat[i,j,2]]
                if kmat[i,j,1] != 0 and kmat[i,j,2] == 0:
                    reskmat[i,j] = pl * theta[kmat[i, j, 1]]
                if kmat[i,j,1] == 0 and kmat[i,j,2] != 0:
                    reskmat[i,j] = pl * kinscal[kmat[i,j,2]]
        
        if not (special or nocolsums):
            for i in range(reskmat.shape[0]):
                for j in range(reskmat.shape[1]):
                    if i != j:
                        reskmat[j,j] = reskmat[j,j]-reskmat[i,j]
    else:
        idx = np.diag_indices(kmat.shape[0])
        kmat[idx] = kmat.sum(axis=0)
        reskmat = kmat
   
    return reskmat


def fullKF(theta, kinscal, kmat, jvec,fixedkmat=False,\
           kinscalspecial = np.empty((0,)), kinscalspecialspec = np.empty((0,)),\
           nocolsums = False):
   kshape = kmat.shape[0]
   A = np.empty((kshape, kshape), dtype=np.float64)
   K = fillK(theta, kinscal, kmat, fixedkmat, kinscalspecial, \
            kinscalspecialspec, nocolsums)
   v, V = scipy.linalg.eig(K)
   gamma = np.matmul(scipy.linalg.inv(V), jvec)
   for i in range(kshape):
       A[i,:] = V[:,i] * gamma[i]
   return(A, v)

def splitVector(params, n_k, irf, disp, n_m, n_d):
    mu = np.nan
    delta = np.nan
    mu_disp = np.empty((0,))
    delta_disp = np.empty((0,))
    k = np.asarray(params[:n_k])
    if irf:
        idx = n_k
        mu = params[idx]
        idx += 1
        delta = params[idx]
        if disp:
            idx += 1
            if n_m != 0:
                mu_disp = np.asarray(params[idx: idx + n_m])
                idx += n_m
            if n_d != 0:
                delta_disp = np.asarray(params[idx: idx + n_d])
            
    return (k, mu, delta, mu_disp, delta_disp)


#def compModel(params, PSI, times, wavenum, n_k, irf=False, disp=False,\
#            n_m=0, n_d=0, fullk=False, kmat=np.empty((0,)),\
#            kinscal=np.empty((0,)), jvec=np.empty((0,)),\
#            fixedkmat=False, kinscalspecial=np.empty((0,)), \
#            kinscalspecialspec=np.empty((0,)), nocolsums=False):
#    (mu_0, delta_0, l_c_mu, l_c_delta, k, mu_disp, delta_disp) =\
#    splitVector(params, irf, disp, n_m, n_d)
#    
#    if fullk:
#        eig = fullKF(k, kinscal, kmat, jvec, fixedkmat, kinscalspecial,\
#                    kinscalspecialspec, nocolsums)
#        k = eig[2]
#        A = eig[1]
#    
#    return calculateC(k, times, irf, mu_0, delta_0, mu_disp, delta_disp, l, l_c_mu, l_c_delta)
     
def compModel(kinpar, times, wavenum, irf=False, mu=0.0, delta=0.0,\
                fullk=False, kmat=np.empty(0,), kinscal=np.empty(0,),\
                jvec=np.empty(0,), fixedkmat=False,kinscalspecial=np.empty(0,), \
                kinscalspecialspec=np.empty(0,), nocolsums=False):
    
    C = np.empty((times.shape[0], kinpar.shape[0]), dtype=np.float64)
    
    if fullk:
       eig = fullKF(kinpar, kinscal, kmat, jvec, fixedkmat,\
            kinscalspecial, kinscalspecialspec, nocolsums)
       k = eig[1]
       #A = eig[0]
    else:
       k = kinpar
   
    if irf:
       calculateCirf(C, k, times, mu, delta)
    else:
       calculateC(C, k, times)
       
    return C
    

def solve(params, PSI, times, wavenum, n_k, irf=False, disp=False, l_c=0,\
            n_m=0, n_d=0, fullk=False, kmat=np.empty((0,)),\
            kinscal=np.empty((0,)), jvec=np.empty((0,)),\
            fixedkmat=False, kinscalspecial=np.empty((0,)), \
            kinscalspecialspec=np.empty((0,)), nocolsums=False):
    res = np.empty(PSI.shape, dtype=np.float64)
    
    (k, mu_0, delta_0, mu_disp, delta_disp)=\
        splitVector(params, n_k, irf, disp, n_m, n_d)

    for i in range(PSI.shape[1]):
#        C = np.empty((times.shape[0], k.shape[0]), dtype=np.float64)
#        if irf:
#            if disp:
#                mu = disp(mu_0, mu_disp, wavenum[i], l_c_mu)
#                delta = disp(delta_0, delta_disp, wavenum[i], l_c_delta)
#            else:
#                mu = mu_0
#                delta = delta_0
#            
#            calculateCirf(C, k, times, mu, delta)
#        else:
#            calculateC(C, k, times)
        if irf:
            if disp:
                mu = irfparF(mu_0, mu_disp, wavenum[i], l_c)
                delta = irfparF(delta_0, delta_disp, wavenum[i], l_c)
            else:
                mu = mu_0
                delta = delta_0
            C = compModel(k, times, wavenum, irf, mu, delta) #Needs to be expanded
        else:
            C = compModel(k, times, wavenum)
        
        b = PSI[:,i]
        res[:,i] = qr(C, b)
        #res[:,i] = scipy.optimize.nnls(C, b)[1]

    return res.flatten()


def main():
    times1 = np.asarray(np.arange(-0.5, 9.98, 0.02))
    times2 = np.asarray(np.arange(0, 1500, 3))
    times = np.hstack((times1, times2))
    #times = np.asarray(np.arange(50, 350, 0.6))
    wavenum = np.asarray(np.arange(12820, 15120, 4.6))
    #wavenum = np.asarray(np.arange(18000, 28000, 20))
    location = np.asarray([14705, 13513, 14492, 14388, 14184, 13986])
    #location = np.asarray([26000, 24000, 20000])
    d = np.asarray([400, 1000, 300, 200, 350, 330])
    #d = np.asarray([2000, 3000, 4000])
    amp = np.asarray([1, 0.2, 1, 1, 1, 1])
    #amp = np.asarray([1, 2, 3])
    
    kinpar = np.asarray([.006667, .006667, 0.00333, 0.00035, 0.0303, 0.000909])
    #kinpar = np.asarray([.01, .05, .05])
    
    mu_0 = -0.05
    delta_0 = 0.02
    
    mu_disp = np.asarray([.001, .001])
    delta_disp = np.asarray([.002, .002])
    
    l_c = 15000
    
    E = np.empty((wavenum.size, location.size), dtype=np.float64)

    for i in range(location.size):
        E[:, i] = amp[i] * np.exp(-np.log(2) * np.square(2 * (wavenum - location[i]) / d[i]))
    
    irf = True
    disp = True
    
    PSI = np.empty((times.size, wavenum.size), dtype=np.float64)
    for i in range(wavenum.size):
        if irf:
            if disp:
                mu = irfparF(mu_0, mu_disp, wavenum[i], l_c)
                delta = irfparF(delta_0, delta_disp, wavenum[i], l_c)
            else:
                mu = mu_0
                delta = delta_0
            C = compModel(kinpar, times, wavenum, irf, mu, delta) #Needs to be expanded
        else:
            C = compModel(kinpar, times, wavenum)
        PSI[:,i] = np.matmul(C, E[i,:])
    

    start_kinpar = np.asarray([.005, 0.003, 0.00022, 0.0300, 0.000888])
    #start_kinpar = np.asarray([.015, 0.04])
    params = start_kinpar
    
    if irf:
        start_irfvec = np.asarray([0.0, 0.1])
        params = np.hstack((params, start_irfvec))
        if disp:
            mu_disp_start = np.asarray([.0015, .0015])
            delta_disp_start = np.asarray([.0015, .0015])
            params = np.hstack((params, mu_disp_start, delta_disp_start))
        
        
    start = time.perf_counter()

    #res = scipy.optimize.least_squares(solve, params, args=(PSI, times, wavenum,\
    #start_kinpar.size, irf, disp, l_c, mu_disp.size), verbose=0, method='trf')
    res = scipy.optimize.leastsq(solve, params, args=(PSI, times, wavenum,\
        start_kinpar.size, irf, disp, l_c, mu_disp.size, delta_disp.size), full_output=0)

    stop = time.perf_counter()

    print(stop - start)
    print(res)

if __name__ == '__main__':
    main()