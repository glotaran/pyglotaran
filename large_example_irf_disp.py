import numpy as np
import scipy.optimize
import scipy.linalg.lapack as lapack
from scipy.special import erf
import time

def calculateC(k, T, mu, tau):
    tau_tilde = tau / (2 * np.sqrt(2 * np.log(2)))
    C = .5* np.exp(np.outer(T, -k))
    term1 = np.exp(np.outer(k, (mu + k * tau_tilde * tau_tilde / 2)))
    term2 = 1 + erf()
    C = np.dot(C, term1)
    return C

def qr(a, c):
    qr, tau, _, _ = lapack.dgeqrf(a, overwrite_a=1)
    c, _, _ = lapack.dormqr("L", "T", qr, tau, c, max(1, a.shape[1]), overwrite_c=1)
    for i in range(a.shape[1]):
        c[i] = 0
    c, _, _ = lapack.dormqr("L", "N", qr, tau, c, max(1, a.shape[1]), overwrite_c=1)
    return c


def solve(params, PSI, times, n_k):
    res = np.empty(PSI.shape, dtype=np.float64)
    k = np.asarray(params[:n_k])
    mu = params[n_k]
    tau = params[n_k + 1]
    C = calculateC(k, times, mu, tau) #This is just an optimization for wavelength independent C matrices
    for i in range(PSI.shape[1]):
        b = PSI[:,i]
        res[:,i] = qr(C, b)

    return res.flatten()


def main():
    times1 = np.asarray(np.arange(-0.5, 9.98, 0.02))
    times2 = np.asarray(np.arange(0, 1500, 3))
    times = np.hstack((times1, times2))
    wavenum = np.asarray(np.arange(12820, 15120, 4.6))
    irfvec = np.asarray([-0.02, 0.05])
    location = np.asarray([14705, 13513, 14492, 14388, 14184, 13986])
    delta = np.asarray([400, 1000, 300, 200, 350, 330])
    amp = np.asarray([1, 0.2, 1, 1, 1, 1])
    kinpar = np.asarray([.006667, .006667, 0.00333, 0.00035, 0.0303, 0.000909])

    E = np.empty((wavenum.size, location.size), dtype=np.float64)

    for i in range(location.size):
        E[:, i] = amp[i] * np.exp(-np.log(2) * np.square(2 * (wavenum - location[i]) / delta[i]))

    C = calculateC(kinpar, times, irfvec[0], irfvec[1])

    PSI = np.dot(C, np.transpose(E))

    start_kinpar = np.asarray([.005, 0.003, 0.00022, 0.0300, 0.000888])
    start_irfvec = np.asarray([0.0, 0.1])

    params = np.hstack((start_kinpar, start_irfvec))

    start = time.perf_counter()

    res = scipy.optimize.least_squares(solve, params, args=(PSI, times, start_kinpar.shape[0]), verbose=0, method='trf')

    stop = time.perf_counter()

    print(stop - start)
    print(res.x)


if __name__ == '__main__':
    main()