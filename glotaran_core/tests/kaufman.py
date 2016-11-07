#  (1) Construct the G matrix
#  (2) Construct the nonzero columns of the D matrix as the derivatives of each
#  column of G with respect
#  to the nonlinear paramters as exemplified in (19)
#  (3) Construct the QR decomposition given in (8) using LAPACK’s DGEQRF
#  (4) Apply the Q matrix from (8) to Y using LAPACK’s DORMQR.
#  (5) Apply the Q matrix from (8) to D using DORMQR and let W represent the last
#  m − u rows of
#  the result.
#  (6) Determine aˆi for i = 1, 2, ··· , s by backsolving the first u elements of
#  QY using R in (8) perhaps
#  using LAPACK’s DTRTRS.
#  (7) Concatenate the last m − u rows of QY to make the residual vector in (20).
