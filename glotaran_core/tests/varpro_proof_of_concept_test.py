import numpy as np
import scipy.optimize
import scipy.linalg.lapack as lapack
import time

def calculateC(k, T):
    return np.exp(np.outer(T, -k))


def qr(a, c):
    qr, tau, _, _ = lapack.dgeqrf(a, overwrite_a=1)
    c, _, _ = lapack.dormqr("L", "T", qr, tau, c, max(1, a.shape[1]), overwrite_c=1)
    for i in range(a.shape[1]):
        c[i] = 0
    c, _, _ = lapack.dormqr("L", "N", qr, tau, c, max(1, a.shape[1]), overwrite_c=1)
    return c


def solve(k, PSI, times):
    res = np.empty(PSI.shape, dtype=np.float64)
    C = calculateC(k, times)
    for i in range(PSI.shape[1]):
        b = PSI[:,i]
        res[:,i] = qr(C, b)

    print(np.sum(res))

    return res.flatten()


def main():

    times = np.asarray(np.arange(0, 1500, 1.5))
    wavenum = np.asarray(np.arange(12820, 15120, 4.6))
    #  wavenum = np.asarray(np.arange(12820, 15120, 4.6))
    location = np.asarray([14705, 13513, 14492, 14388, 14184, 13986])
    delta = np.asarray([400, 1000, 300, 200, 350, 330])
    amp = np.asarray([1])
    kinpar = np.asarray([.006667])
    #  kinpar = np.asarray([.006667, .006667, 0.00333, 0.00035, 0.0303, 0.000909])

    E = np.empty((1, 1), dtype=np.float64, order="F")

    for i in range(amp.size):
        E[:, i] = amp[i]
        #  E[:,i] = amp[i] * np.exp(-np.log(2) * np.square(2 * (wavenum - location[i])/delta[i]))

    C = calculateC(kinpar, times)

    PSI = np.dot(C, np.transpose(E))
    print(PSI.shape)

    start_kinpar = np.asarray([.005])
    #  start_kinpar = np.asarray([.005, 0.003, 0.00022, 0.0300, 0.000888])

    start = time.perf_counter()

    res = scipy.optimize.least_squares(solve, start_kinpar, args=(PSI, times),
                                       verbose=2, method='lm', gtol=1e-5)

    stop = time.perf_counter()

    print(stop - start)
    print(res.x)


if __name__ == '__main__':
    main()
