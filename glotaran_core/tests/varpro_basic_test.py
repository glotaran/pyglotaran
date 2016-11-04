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


def solve(k, single_trace, times):
    res = np.empty(single_trace.shape, dtype=np.float64)
    C = calculateC(k, times)
    assert len(single_trace.shape)==1, "data is not a single trace"
        
    res = qr(C, single_trace)

    return res.flatten()


def main():

    times = np.asarray(np.arange(0, 1500, 1.5))
    amp = np.asarray([1,2])
    kinpar = np.asarray([.006667,0.0303])

    C = calculateC(kinpar, times)

    single_trace = np.dot(C, amp)

    start_kinpar = np.asarray([.005, 0.02])
    start = time.perf_counter()

    res = scipy.optimize.least_squares(solve, start_kinpar, args=(single_trace, times), verbose=3, method='lm')

    stop = time.perf_counter()

    print(stop - start)
    print(res.x)


if __name__ == '__main__':
    main()
