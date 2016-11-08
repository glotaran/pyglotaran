import numpy as np
import scipy.optimize
import scipy.linalg.lapack as lapack
import time

def calculateC(k, T):
    return np.exp(np.outer(T, -k))


def qr(a, c):    
    # TODO: audit this routine
    qr, tau, _, _ = lapack.dgeqrf(a, overwrite_a=1)
    c, _, _ = lapack.dormqr("L", "T", qr, tau, c, max(1, a.shape[1]), overwrite_c=1)
    for i in range(a.shape[1]):
        c[i] = 0
    c, _, _ = lapack.dormqr("L", "N", qr, tau, c, max(1, a.shape[1]), overwrite_c=1)    
    return c


def solve(k, single_trace, times):
    if len(single_trace.shape) == 1:
        single_trace.shape = (-1, 1)
    res = np.empty(single_trace.shape, dtype=np.float64)
    C = calculateC(k, times)            
    for i in range(single_trace.shape[1]):
        b = single_trace[:, i]
        res[:, i] = qr(C, b)
    r = np.sum(res)
    print("ITER: {}".format(C.shape))
    print("ITER: {}".format(np.sum(C)))
    print("ITER: {}".format(r*r))
    print("ITER: {}".format(res.shape))
    print("ITER: {}".format(res.flatten().shape))
    return res.flatten()


def main():

    times = np.asarray(np.arange(0, 1500, 1.5))
    amp = np.asarray([[1, 2]])
    print(amp.shape)
    kinpar = np.asarray([0.0101,.00202])

    C = calculateC(kinpar, times)

    single_trace = np.dot(C, np.transpose(amp))
    print(single_trace.shape)
    #  assert len(single_trace.shape)==1, "data is not a single trace"

    start_kinpar =  0.1*kinpar
    start = time.perf_counter()

    res = scipy.optimize.least_squares(solve, start_kinpar, args=(single_trace, times), verbose=2, method='lm')
    #TODO: also get amplitudes from varpro out, not just parameter estimates
    stop = time.perf_counter()

    print(stop - start)
    print(res.x)


if __name__ == '__main__':
    main()
