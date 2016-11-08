import scipy.linalg.lapack as lapack


def qr_decomposition(a, c):
    qr, tau, _, _ = lapack.dgeqrf(a, overwrite_a=0)
    c, _, _ = lapack.dormqr("L", "T", qr, tau, c, max(1, a.shape[1]), overwrite_c=0)
    for i in range(a.shape[1]):
        c[i] = 0
    c, _, _ = lapack.dormqr("L", "N", qr, tau, c, max(1, a.shape[1]), overwrite_c=0)
    return c
