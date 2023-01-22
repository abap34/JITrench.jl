using JITrench
using Printf

f(x) = x^4 - 2x^2

x = Scalar(2.0)
iters = 10

for i in 1:iters    
    y = f(x)
    @printf "[iter] %5i | [y] %.7f | [x] %.7f \n" i y.values x.values
    
    JITrench.AutoDiff.cleargrad!(x)
    backward!(y, create_graph=true)
    gx = x.grad
    JITrench.AutoDiff.cleargrad!(gx)
    backward!(gx)
    gx2 = x.grad
    x.values -= gx.values / gx2.values
end

