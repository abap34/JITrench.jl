using JITrench 
using Random
using Printf
using Plots

Random.seed!(10)

ENV["GKSwstype"] = "nul"

function generate_datset(N)
    x = rand(N, 1) 
    y = sin.(2π .* x) .+ (rand(N, 1) .* 0.5)
    return x,  y
end

function mse(y_true, y_pred)
    diff = y_true .- y_pred
    return JITrench.sum(diff .^ 2) / length(diff.values)
end

function predict(x, l1, l2)
    y = l1(x)
    y = JITrench.sigmoid.(y)
    y = l2(y)
    return y
end

function train(x, y, hidden_dim, out_dim, n_iter; lr=1e-1, log_interval=100)
    anim = Animation()
    x = Variable(x)
    y = Variable(y)
    history = []
    l1 = JITrench.Linear(hidden_dim)
    l2 = JITrench.Linear(out_dim)
    for iter in 1:n_iter
        y_pred = predict(x, l1, l2)
        loss = mse(y, y_pred)
        if iter == 1
            JITrench.plot_graph(loss, to_file="plot.png")
        end
        push!(history, loss.values)

        JITrench.cleargrads!(l1)
        JITrench.cleargrads!(l2)
        backward!(loss)
        for param in values(l1.param._dict)
            param.values -= lr * param.grad.values
        end
        for param in values(l2.param._dict)
            param.values -= lr * param.grad.values
        end

        if (iter - 1) % log_interval == 0
            @printf "iters %4i [loss] %.2f\n" iter loss.values 
            _x = reshape(collect(0:0.01:1), :, 1)
            y_pred = predict(Variable(_x), l1, l2).values
            plt = scatter(x.values, y.values, label="data")
            plot!(plt, 0:0.01:1, y_pred, title="[mse : $(loss.values)]", label="predict")
            frame(anim, plt)
        end
    end 
    return  history, anim
end

x, y = generate_datset(100)

input_dim = 1
hidden_dim = 10
out_dim = 1
n_iters = 10000

history, anim = train(x, y, hidden_dim, out_dim, n_iters)

println("finish train.")

gif(anim, "fitting_history.gif", fps=10)

plot(history, title="learning curve", yaxis=:log10)
savefig("curve.png")