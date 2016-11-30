import scipy.linalg.lapack as lapack


def qr_decomposition(A, B):
    qr, tau, _, _ = lapack.dgeqrf(A, overwrite_a=0)
    B, _, _ = lapack.dormqr("L", "T", qr, tau, B, max(1, A.shape[1]),
                            overwrite_c=0)
    for i in range(A.shape[1]):
        B[i] = 0
    B, _, _ = lapack.dormqr("L", "N", qr, tau, B, max(1, A.shape[1]),
                            overwrite_c=0)
    return B


def qr_decomposition_coeff(A, B):
    qr, tau, _, _ = lapack.dgeqrf(A, overwrite_a=0)
    B, _, _ = lapack.dormqr("L", "T", qr, tau, B, max(1, A.shape[1]),
                            overwrite_c=0)
    for i in range(A.shape[1]):
        B[i] = 0
    B, _, _ = lapack.dormqr("L", "N", qr, tau, B, max(1, A.shape[1]),
                            overwrite_c=0)
    B, _ = lapack.dtrtrs(qr, B, 0, 0, 0, max(1, B.shape[0]), 0)
    return B
