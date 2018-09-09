import scipy.linalg.lapack as lapack


def qr_residual(A, B):

    # Kaufman Q2 step 3
    qr, tau, _, _ = lapack.dgeqrf(A)

    # Kaufman Q2 step 4

    B, _, _ = lapack.dormqr("L", "T", qr, tau, B, max(1, A.shape[1]),
                            overwrite_c=0)

    for i in range(A.shape[1]):
        B[i] = 0

    # Kaufman Q2 step 5

    B, _, _ = lapack.dormqr("L", "N", qr, tau, B, max(1, A.shape[1]),
                            overwrite_c=1)
    return B


def qr_coefficents(A, B):

    # Kaufman Q2 step 3

    qr, tau, _, _ = lapack.dgeqrf(A)

    # Kaufman Q2 step 4

    B, _, _ = lapack.dormqr("L", "T", qr, tau, B, max(1, A.shape[1]))

    # Kaufman Q2 step 6

    P, _ = lapack.dtrtrs(qr, B)
    return P
