using JITrench
using Printf

rosenbrock(x₀, x₁) = 100 * (x₁ - x₀^2)^2 + (x₀ - 1)^2

x₀ = Scalar(0.0)
x₁ = Scalar(2.0)
lr = 1e-3
iters = 10000

for i in 1:iters    
    y = rosenbrock(x₀, x₁)
    if i % 100 == 0
        @printf "[iter] %5i | [y] %.7f | [x₀, x₁] %.7f, %.7f \n" i y.values x₀.values x₁.values
    end

    JITrench.AutoDiff.cleargrad!(x₀)
    JITrench.AutoDiff.cleargrad!(x₁)
    backward!(y)
    
    x₀.values -= lr * x₀.grad
    x₁.values -= lr * x₁.grad
end
