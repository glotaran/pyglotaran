# a kinetic model is evaluated 

shared_times <- c(head(seq(10, 50, by=1.5),-1),
                  head(seq(50, 1000, by=15),-1),
                  head(seq(1000, 3100, by=100),-1))

times_no_IRF <- c(
           head(seq(0, 10, by=0.01),-1),
           shared_times
         )

times_with_IRF <- c(head(seq(-10, -1, by=0.1),-1),
           head(seq(-1, 10, by=0.01),-1),
           shared_times)

print(sprintf("length(times) = %i ", length(times_no_IRF)))
print(sprintf("length(times) = %i ", length(times)))