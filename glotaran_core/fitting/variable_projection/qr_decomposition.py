import scipy.linalg.lapack as lapack


def qr_decomposition(A, B):

    # Kaufman Q2 step 3
    qr, tau, _, _ = lapack.dgeqrf(A)

    # Kaufman Q2 step 4

    B, _, _ = lapack.dormqr("L", "T", qr, tau, B, max(1, A.shape[1]))
    for i in range(A.shape[1]):
        B[i] = 0

    # Kaufman Q2 step 5

    B, _, _ = lapack.dormqr("L", "N", qr, tau, B, max(1, A.shape[1]))

    return B


def qr_decomposition_coeff(A, B):

    # Kaufman Q2 step 3

    qr, tau, _, _ = lapack.dgeqrf(A)

    # Kaufman Q2 step 4

    B, _, _ = lapack.dormqr("L", "T", qr, tau, B, max(1, A.shape[1]))

    # Q, _, _ = lapack.dorgqr(qr, tau)

    #  for i in range(A.shape[1]):
    #      B[i] = 0
    # Kaufman Q2 step 5

    #  B, _, _ = lapack.dormqr("L", "N", qr, tau, B, max(1, A.shape[1]),
    #                          overwrite_c=0)

    # Kaufman Q2 step 6
    P, _ = lapack.dtrtrs(qr, B)
    # Q.shape = (Q.shape[0],)
    # print(qr.shape, B.shape)
    return P
