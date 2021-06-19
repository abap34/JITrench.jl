using Random
using Printf
using JITrench


f(x) = x^4 - 2x^2

x = Variable(2.0)

N_ITERS = 10

for i in 1:N_ITERS
    @printf "[%.02i] x = %.5f | f(x) = %.5f\n" i x.values f(x).values
    y = f(x)
    cleargrad!(x)
    backward!(y)
    gx = x.grad
    cleargrad!(x)
    backward!(gx)
    gx2 = x.grad
    x.values -= gx.values / gx2.values
end