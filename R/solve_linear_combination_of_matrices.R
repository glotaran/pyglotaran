# Lets start with 3 2x2 matrices,
# A = [2,3;
#      5,7]
# B = [11,13;
#      17,19]
# C = [23,29;
#      31,37]
# But lets vectorize them.
A <- c(2,3,5,7)
B <- c(11,13,17,19)
C <- c(23,29,31,37)
abc_matrix <- matrix(c(A,B,C),4,3)

# Now lets define the amplitude vector (which we want to recover):
x <- c(-1,3,7) #x1 = -1; x2 = 3; x3 = 7;

# Now lets generate the input matrix, what we will start with later:
z_matrix <- abc_matrix%*%x

# Now, knowing A,B,C and the z_matrix, recover the amplitude vector
x_recov <- qr.coef(qr(abc_matrix), z_matrix)
# Print the answer to the terminal
print(x_recov)

################
## works
###############

z_matrix2 <- as.vector( (A * -1) + (B * 3) + (C * 7) )

x_recov2 <- qr.coef(qr(abc_matrix), z_matrix2)

################

# Now do it for 3 3x3 matrices,
A <- c(2,3,5,7,11,13,17,19,23)
B <- c(29,31,37,41,43,43,47,53,61)
C <- c(67,71,73,79,83,89,97,101,103)
abc_matrix <- matrix(c(A,B,C),9,3)

# Now lets define the amplitude vector (which we want to recover):
x <- c(-1,3,7) #x1 = -1; x2 = 3; x3 = 7;

# Now lets generate the input matrix, what we will start with later:
z_matrix <- abc_matrix%*%x

# Now, knowing A,B,C and the z_matrix, recover the amplitude vector
x_recov <- qr.coef(qr(abc_matrix), z_matrix)
# Print the answer to the terminal
print(x_recov)

################
## works
################

z_matrix2 <- as.vector( (A * -1) + (B * 3) + (C * 7) )

x_recov2 <- qr.coef(qr(abc_matrix), z_matrix2)

################
# works
################

A <- matrix(rnorm(16), nrow=4,ncol=4) 
B <- matrix(rnorm(16), nrow=4,ncol=4) 
C <- matrix(rnorm(16), nrow=4,ncol=4) 
D <- matrix(rnorm(16), nrow=4,ncol=4) 

allM <- cbind(as.vector(A),as.vector(B),as.vector(C),as.vector(D)) 

d <- as.vector( (A * .4) + (B * -.1) + (C * 4) + (D * 19 ))

x_recov <- qr.coef(qr(allM), d)

################
# works
################

A <- matrix(rnorm(25), nrow=5,ncol=5) 
B <- matrix(rnorm(25), nrow=5,ncol=5) 
C <- matrix(rnorm(25), nrow=5,ncol=5) 
D <- matrix(rnorm(25), nrow=5,ncol=5) 
E <- matrix(rnorm(25), nrow=5,ncol=5)
allM <- cbind(as.vector(A),as.vector(B),as.vector(C),as.vector(D),as.vector(E)) 

d <- as.vector( (A * 18) + (B * -.1) + (C * 4) + (D * 19 ) + E * -.4)

x_recov <- qr.coef(qr(allM), d)


