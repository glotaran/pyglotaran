import numpy as np
import scipy.optimize
import scipy.linalg.lapack as lapack
import time

from calculateC import calculateC


def qr(a, c):
    qr, tau, _, _ = lapack.dgeqrf(a, overwrite_a=1)
    c, _, _ = lapack.dormqr("L", "T", qr, tau, c, max(1, a.shape[1]), overwrite_c=1)
    for i in range(a.shape[1]):
        c[i] = 0
    c, _, _ = lapack.dormqr("L", "N", qr, tau, c, max(1, a.shape[1]), overwrite_c=1)
    return c


def solve(params, PSI, times, wavenum, n_k, irf=False, disp=False, n_m=0, n_d=0):
    res = np.empty(PSI.shape, dtype=np.float64)
    mu_0 = np.nan
    delta_0 = np.nan
    mu_disp = np.empty((0,))
    delta_disp = np.empty((0,))
    l = np.nan
    l_c_mu = np.nan
    l_c_delta = np.nan
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
            l = wavenum[i]
        C = calculateC(k, times, irf, mu_0, delta_0, mu_disp, delta_disp, l, l_c_mu, l_c_delta)
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

    C = calculateC(kinpar, times, True, irfvec[0], irfvec[1], np.empty((0,)), np.empty((0,)), np.nan, np.nan, np.nan)

    PSI = np.dot(C, np.transpose(E))

    start_kinpar = np.asarray([.005, 0.003, 0.00022, 0.0300, 0.000888])
    start_irfvec = np.asarray([0.0, 0.1])

    params = np.hstack((start_kinpar, start_irfvec))

    start = time.perf_counter()

    res = scipy.optimize.least_squares(solve, params, args=(PSI, times, wavenum, start_kinpar.shape[0], True), verbose=0, method='trf')

    stop = time.perf_counter()

    print(stop - start)
    print(res.x)


if __name__ == '__main__':
    main()