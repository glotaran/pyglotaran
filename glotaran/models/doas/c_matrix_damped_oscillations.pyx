
import cython
cimport cython

import numpy as np
cimport numpy as np

cdef extern from "complex.h":
    double complex exp(double complex)


def __init__():
    np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
def calculateC(double[:, :] C, idxs, double[:] f, double[:] k, double[:] T, double scale):
    nr_times = T.shape[0]
    nr_comps = k.shape[0]
    cdef int n_c, n_t, n_k
    cdef double t_n, k_n
    for n_k in range(nr_comps):
        n_c_sin = idxs[n_k]
        n_c_cos = idxs[n_k] + 1
        k_n = k[n_k]
        f_n = f[n_k]
        for n_t in range(nr_times):
            t_n = T[n_t]
            osc = scale * exp(-k_n * t_n - 1j * f_n *t_n)
            print("fff")
            print(f_n)
            print(t_n)
            print(f_n*t_n)
            print(exp(-1j* f_n*t_n))
            C[n_t, n_c_sin] += np.real(osc)
            C[n_t, n_c_cos] += np.imag(osc)
