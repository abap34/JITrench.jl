using JITrench
using Printf
using Plots

rosenbrock(x₀, x₁) = 100 * (x₁ - x₀^2)^2 + (x₀ - 1)^2

x₀ = Scalar(0.0)
x₁ = Scalar(5.0)
lr = 1e-3
iters = 10000

x₀_history = Float64[]
x₁_history = Float64[]

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

    push!(x₀_history, x₀.values)
    push!(x₁_history, x₁.values)
end

default(legend=false)

x = -3:0.1:3
y = -1:0.1:5

anim = Animation()

for i in 1:13
    p = plot(x, y, rosenbrock, st=:surface, alpha=0.8)
    plot!(p, camera=(10 * (1 + cos(log2(i))), 40))
    scatter!(x₀_history[2^(i-1):2^i], x₁_history[2^(i-1):2^i], rosenbrock.(x₀_history[2^(i-1):2^i], x₁_history[2^(i-1):2^i]))
    frame(anim, p)
end

gif(anim, "example/visualize/gradient_decent.gif", fps=4)