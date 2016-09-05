import numpy as np
import lmfit
import scipy
import scipy.lapack as lapack
from calculateC import calculateC, calculateCirf

def create_k_matrix(values, indizes):
    filt = np.vectorize(lambda i: values[i])
    k_matrix = filt(np.asarray(indizes))
    return k_matrix.reshape(indizes.shape)
    
def full_k_matrix(values, indizes, j_vector):
    A = np.empty(indizes.shape, dtype=np.float64)
    K = create_k_matrix(values, indizes)
    eigenvalues, eigenvectors = np.linalg.eig(K.astype(np.float64))
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    gamma = np.matmul(scipy.linalg.inv(eigenvectors), j_vector)
    for i in range(indizes.shape[0]):
        A[i,:] = eigenvectors[:,i] * gamma[i]
    return(A, eigenvalues)

def irfparF(par_0, par_disp=np.empty((0,)), l=0, l_c=0):
    disp_s = par_0
    for i in range(par_disp.size):
        disp_s += par_disp[i] * np.power(((l - l_c) / 100), i + 1)
    return disp_s

def qr(a, c):    
    qr, tau, _, _ = lapack.dgeqrf(a, overwrite_a=1)
    c, _, _ = lapack.dormqr("L", "T", qr, tau, c, max(1, a.shape[1]), overwrite_c=1)
    for i in range(a.shape[1]):
        c[i] = 0
    c, _, _ = lapack.dormqr("L", "N", qr, tau, c, max(1, a.shape[1]), overwrite_c=1)
    return c

class Model(lmfit.Model):
    def __init__(self, file):
        with f = open(file, "r"):
            self.parameters = lmfit.Parameters()
            self.datasets = {}
            self.compartments
            docs = list(yaml.load_all(f))
            for doc in docs:
                self.datasets = 
    
    
    def eval(self, dataset=None, **kwargs):
        try:
            timestamps = kwargs["timestamps"]
            wavelengths = kwargs["wavelengths"]
            location = kwargs["location"]
            amp = kwargs["amp"]
            delta = kwargs["delta"]            
            E = np.empty((wavelengths.size, location.size), dtype=np.float64)
            for i in range(location.size):
                E[:, i] = amp[i] * np.exp(-np.log(2) * np.square(2 * (wavelengths - location[i]) / delta[i]))
            
            for i in range(wavelengths.size):
                C = self.calcC(wavelenghts[i], **kwargs)
                PSI[:,i] = np.dot(scale * C, E[i,:])
                        
            if isinstance(dataset, dict):
                dataset["timestamps"] = timestamps
                dataset["wavelengths"] = wavelenghts
                dataset["PSI"] = PSI
            return PSI
            
        except KeyError as e:
            pass
                        
    def calcC(self, l, **kwargs):
        timestamps = kwargs["timestamps"]
        A = None
        if self.j_parameters:
            indizes = kwargs["indizes"]
            (A, k) = full_k_matrix(list(self.k_parameters.values()),\
                indizes, list(self.j_parameters.values()))
        else:
            k = np.asarray(list(self.k_parameters.values()), dtype=np.float64)

        C = np.empty((timestamps.shape[0], k.shape[0]), dtype=np.float64)        
        
        if self.irf_parameters:
            mu = self.irf_parameters["mu"]
            delta = self.irf_parameters["delta"]
            if self.mu_dispersion_parameters:
                l_c_mu = kwargs["l_c_mu"]
                mu = irfparF(mu, np.asarray(list(self.mu_dispersion_parameters.values()), dtype=np.float64), l, l_c_mu)
            if self.delta_dispersion_parameters:
                l_c_delta = kwargs["l_c_delta"]
                delta = irfparF(delta, np.asarray(list(self.delta_dispersion_parameters.values()), dtype=np.float64), l, l_c_delta)
            C = calculateCirf(C, timestamps, k, mu, delta)
        else:
            C = calculateC(C, timestamps, k)
        
        if A:
            C = np.dot(C, A)
        
        return C
        
    def _residual(self, params, data, weights, **kwargs):
        for i in range(data[i]):
            if self.irf_parameters:
                mu = self.irf_parameters["mu"]
                delta = self.irf_parameters["delta"]
                if self.mu_dispersion_parameters:
                    l_c_mu = kwargs["l_c_mu"]
                    mu = irfparF(mu, np.asarray(list(self.mu_dispersion_parameters.values()), dtype=np.float64), l, l_c_mu)
                if self.delta_dispersion_parameters:
                    l_c_delta = kwargs["l_c_delta"]
                    delta = irfparF(delta, np.asarray(list(self.delta_dispersion_parameters.values()), dtype=np.float64), l, l_c_delta)
                C = calculateCirf(C, timestamps, k, mu, delta)
            else:
                C = calculateC(C, timestamps, k)
            
            b = data[:,i]
            res[:,i] = qr(C, b)
            return res.flatten()