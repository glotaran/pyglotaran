import numpy as np
import scipy.optimize
import scipy.linalg.lapack as lapack
from scipy.special import erf
import time


def disp(par_0, par_disp=None, l=None, l_c=None):
    disp_s = par_0
    if par_disp:
        if not l or not l_c:
            raise ValueError("Parameters missing")
        for i in range(par_disp.shape[0]):
            disp_s += par_disp[i] * np.power(((l - l_c) / 100), i)
    return disp_s


def calculateC(k, T, mu_0=None, delta_0=None, mu_disp=None, delta_disp=None, l=None, l_c_mu=None, l_c_delta=None):
    C = np.exp(np.outer(T, -k))
    if not mu_0 and not delta_0:
        return C
    mu = disp(mu_0, mu_disp, l, l_c_mu)
    delta = disp(delta_0, delta_disp, l, l_c_delta)
    delta_tilde = delta / (2 * np.sqrt(2 * np.log(2)))
    for i in range(T.shape[0]):
        for j in range(k.shape[0]):
            t_i = T[i]
            k_j = k[j]
            term_1 = np.exp(k_j * (k_j * delta_tilde * delta_tilde / 2))
            term_2 = 1 + erf((t_i - (mu + k_j * delta_tilde * delta_tilde)) / (np.sqrt(2) * delta_tilde))
            C[i, j] *= .5 * term_1 * term_2
    return C


def qr(a, c):
    qr, tau, _, _ = lapack.dgeqrf(a, overwrite_a=1)
    c, _, _ = lapack.dormqr("L", "T", qr, tau, c, max(1, a.shape[1]), overwrite_c=1)
    for i in range(a.shape[1]):
        c[i] = 0
    c, _, _ = lapack.dormqr("L", "N", qr, tau, c, max(1, a.shape[1]), overwrite_c=1)
    return c


def solve(params, PSI, times, n_k, irf=False, disp=False, n_m=0, n_d=0):
    res = np.empty(PSI.shape, dtype=np.float64)
    mu_0 = None
    delta_0 = None
    mu_disp = None
    delta_disp = None
    l = None
    l_c_mu = None
    l_c_delta = None
    k = np.asarray(params[:n_k])
    if irf:
        idx = n_k
        mu_0 = params[idx]
        idx += 1
        delta_0= params[idx]
        if disp:
            l_c_mu = params[idx]
            idx += 1
            mu_disp = np.asarray(params[idx: idx + n_m])

            idx += n_m

            l_c_delta = params[idx]
            idx += 1
            delta_disp = np.asarray(params[idx: idx + n_d])

    for i in range(PSI.shape[1]):
        if disp:
            l = i
        C = calculateC(k, times, mu_0, delta_0, mu_disp, delta_disp, l, l_c_mu, l_c_delta)
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

    res = scipy.optimize.least_squares(solve, params, args=(PSI, times, start_kinpar.shape[0], True), verbose=0, method='trf')

    stop = time.perf_counter()

    print(stop - start)
    print(res.x)


if __name__ == '__main__':
    main()