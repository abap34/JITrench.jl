```@meta
CurrentModule = JITrench
```

# JITrench

JITrench.jl is Lightweight Automatic Differentiation & DeepLearning Framework implemented in pure Julia.

# Quick Tour

## Automatic Differentiation

```julia
julia> using JITrench

julia> f(x) = sin(x) + 1
f (generic function with 1 method)

julia> JITrench.@diff! f(x)
f′ (generic function with 1 method)

julia> f′(π)
-1.0
```

## Train Neural Network

```julia
using JITrench
using JITrench.NN 
using Printf


N = 100
p = 1
n_iter = 20000

x = rand(N, p)
y = sin.(2π .* x) + (rand(N, p) / 1)

function model(x)
    x = NN.Linear(out_dim=10)(x)
    x = NN.functions.sigmoid.(x)
    x = NN.Linear(out_dim=1)(x)
    return NN.result(x)
end

params = NN.init(model, NN.Initializer((nothing, 1)))
optimizer = NN.SGD(params, 1e-1)

x = Tensor(x)
y = Tensor(y)

for iter in 1:n_iter
    pred = NN.apply(model, x, params)
    loss = NN.functions.mean_squared_error(y, pred)
    NN.cleargrads!(params)
    backward!(loss)
    NN.optimize!(optimizer)
    if (iter % 500 == 0)
        @printf "[iters] %4i [loss] %.4f\n" iter loss.values 
    end
end


NN.save_weight(params, "weight")
```

![](https://github.com/abap34/JITrench.jl/blob/master/example/NN/learning.gif)



