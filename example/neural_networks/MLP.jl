import JITrench
using JITrench: Variable, Model, Linear, sigmoid, SGD,  mean_squared_error, cleargrads!, backward!, do_optimize!
using Printf
using Random
using Plots

Random.seed!(10)

ENV["GKSwstype"] = "nul"

function generate_dataset(N)
    x = rand(N, 1) 
    y = sin.(2π .* x) .+ (rand(N, 1) .* 0.5)
    return Variable(x),  Variable(y)
end

function train(model, x, y; n_iters=10000, log_interval=200, lr=1e-1)
    optimizer = SGD(layers, lr=1e-1)
    anim = Animation()
    history = []
    JITrench.plot_model(model, x)
    for iter in 1:n_iters
        y_pred = model(x)
        loss = mean_squared_error(y, y_pred)
        push!(history, loss.values)
        cleargrads!(model, layers, skip_uninit=true)
        backward!(loss)
        do_optimize!(model, optimizer)
        if (iter - 1) % log_interval == 0
            @printf "[iters] %4i [loss] %.2f\n" iter loss.values 
            val = reshape(collect(0:0.01:1), :, 1)
            y_pred = model(val).values
            p1 = scatter(x.values, y.values, label="data")
            plot!(p1, 0:0.01:1, y_pred, title="[mse : $(repr(loss.values)[1:10])]", label="predict")
            p2 = plot(history, title="learning curve", xlims=(0, n_iters))
            plt = plot(p1, p2, size=(800, 1000), layout=(2, 1))
            frame(anim, plt)
        end
    end 
    return history, anim
end    

mutable struct MLP <: Model
    l1 :: Linear
    l2 :: Linear
    MLP(hidden_dim, out_dim) = new(Linear(hidden_dim), Linear(out_dim))
end

layers(model::MLP) = (model.l1, model.l2)

(model::MLP)(x) = x |> model.l1 .|> sigmoid |> model.l2

model = MLP(10, 1)

history, anim = train(model, generate_dataset(100)...)
gif(anim, "fitting_history.gif", fps=8)

