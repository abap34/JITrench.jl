# JITrench


<h1 align="center">
  <img src=https://cdn.discordapp.com/attachments/810478331790491681/855768153913425930/unknown.png  width=450><br/>
</h1>
<p align="center">lightweight, <br>scalable, <br>and affordable deep learning framework.<br>Let's dive into the deep trenches of the loss function <br>with JITrench.jl.</b></p>

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://abap34.github.io/JITrench.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://abap34.github.io/JITrench.jl/dev)
[![Build Status](https://travis-ci.com/abap34/JITrench.jl.svg?branch=master)](https://travis-ci.com/abap34/JITrench.jl)


---

## Install
```
]add https://github.com/abap34/JITrench.jl
```

## Auto Grad
```julia                                                 
julia> using JITrench
[ Info: Precompiling JITrench [573e55e6-eb53-4bcf-a884-6670806246ed]

julia> x = Variable(3)
name: nothing
values: 3
creator: User-Defined (nothing)

julia> f(x) = 2x^2 + 5x + 10
f (generic function with 1 method)

julia> y = f(x)
name: nothing
values: 43
creator: JITrench.Add

julia> backward!(y)

julia> x.grad
name: nothing
values: 17
creator: JITrench.Add
```


# Display a calculation graph

```julia
julia> x = Variable(2.5, name="x")
name: x 
values: 2.5
creator: User-Defined(nothing)

julia> y = Variable(3.5, name="y")
name: y 
values: 3.5
creator: User-Defined(nothing)

julia> goldstain(x, y) = (1 + (x + y + 1)^2 * (19 - 14x + 3x^2 - 14y + 6x*y + 3y^2)) *  (30 + (2x - 3y)^2 * (18 - 32x + 12x^2 + 48y - 36x*y + 27*y^2))
goldstain (generic function with 1 method)

julia> z = goldstain(x, y)
name: nothing 
values: 1.260939725e7
creator: JITrench.Mul

julia> JITrench.plot_graph(z, to_file="graph.png")
```

![](https://media.discordapp.net/attachments/810478331790491681/924093946496434207/graph.png?width=810&height=854)


```julia
julia> x₁ = Variable([1, 2, 3], name="x₁")
name: x₁ 
values: [1, 2, 3]
creator: User-Defined(nothing)

julia> x₂ = Variable([0, 2, 4], name="x₂")
name: x₂ 
values: [0, 2, 4]
creator: User-Defined(nothing)

julia> Δx = x2 .- x1
name: nothing 
values: [1, 0, 1]
creator: JITrench.Broadcasting{typeof(-)}

julia> Δx.name = "Δx"
"Δx"

julia> JITrench.plot_graph(Δx, to_file="graph2.png")
```

![](https://media.discordapp.net/attachments/810478331790491681/924093946844573767/graph2.png?width=810&height=290)

## Example: Train Neural Networks

```julia
using JITrench: Variable, Model, Linear, sigmoid, SGD,  mean_squared_error, cleargrads!, backward!, do_optimize!
using Printf
using Random

Random.seed!(10)

function generate_dataset(N)
    x = rand(N, 1) 
    y = sin.(2π .* x) .+ (rand(N, 1) .* 0.5)
    return Variable(x),  Variable(y)
end

function train(model, x, y; n_iters=10000, log_interval=200, lr=1e-1)
    optimizer = SGD(layers, lr=1e-1)
    JITrench.plot_model(model, x)
    for iter in 1:n_iters
        y_pred = model(x)
        loss = mean_squared_error(y, y_pred)
        cleargrads!(model, layers, skip_uninit=true)
        backward!(loss)
        do_optimize!(model, optimizer)
        if (iter - 1) % log_interval == 0
            @printf "[iters] %4i [loss] %.2f\n" iter loss.values 
        end
    end 
end    

mutable struct MLP <: Model
    l1 :: Linear
    l2 :: Linear
    MLP(hidden_dim, out_dim) = new(Linear(hidden_dim), Linear(out_dim))
end

layers(model::MLP) = (model.l1, model.l2)

(model::MLP)(x) = x |> model.l1 .|> sigmoid |> model.l2

model = MLP(10, 1)

train(model, generate_dataset(100)...)
```


![](https://media.discordapp.net/attachments/810478331790491681/924441367923523624/fitting_history.gif)
![](https://media.discordapp.net/attachments/810478331790491681/924441368514928700/model.png?width=810&height=358)