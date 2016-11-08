import scipy.linalg.lapack as lapack


def r_decomposition(A, B):
    qr, tau, _, _ = lapack.dgeqrf(A, overwrite_a=1)
    B, _, _ = lapack.dormqr("L", "T", qr, tau, B, max(1, A.shape[1]),
                            overwrite_c=1)
    for i in range(A.shape[1]):
        B[i] = 0
    B, _, _ = lapack.dormqr("L", "N", qr, tau, B, max(1, A.shape[1]),
                            overwrite_c=1)
    return B


def qr_decomposition(a, c):
    qr, tau, _, _ = lapack.dgeqrf(a, overwrite_a=0)
    c, _, _ = lapack.dormqr("L", "T", qr, tau, c, max(1, a.shape[1]), overwrite_c=0)
    for i in range(a.shape[1]):
        c[i] = 0
    c, _, _ = lapack.dormqr("L", "N", qr, tau, c, max(1, a.shape[1]), overwrite_c=0)
    return c


