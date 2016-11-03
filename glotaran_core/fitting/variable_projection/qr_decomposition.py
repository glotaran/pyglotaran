import scipy.linalg.lapack as lapack


def qr_decomposition(A, B):
    qr, tau, _, _ = lapack.dgeqrf(A, overwrite_A=1)
    c, _, _ = lapack.dormqr("L", "T", qr, tau, B, max(1, A.shape[1]), overwrite_B=1)
    for i in range(A.shape[1]):
       B[i] = 0
    B, _, _ = lapack.dormqr("L", "N", qr, tau, B, max(1, a.shape[1]), overwrite_c=1)
    return B

