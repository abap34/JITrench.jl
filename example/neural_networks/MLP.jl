import JITrench
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
    for iter in 1:n_iters
        if iter == 1
            cleargrads!(model, layers, skip_uninit=true)
        else
            cleargrads!(model, layers)
        end
        y_pred = model(x)
        loss = mean_squared_error(y, y_pred)
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