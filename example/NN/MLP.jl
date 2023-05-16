using JITrench
using JITrench.NN 
using Plots
using Printf

ENV["GKSwstype"] = "nul"

N = 100
p = 1
n_iter = 20000

x = rand(N, p)
y = sin.(2π .* x) + (rand(N, p) / 1)
p = scatter(x, y)

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


history = Float64[]
anim = Animation()

for iter in 1:n_iter
    pred = NN.apply(model, x, params)
    loss = NN.functions.mean_squared_error(y, pred)
    push!(history, loss.values)
    NN.cleargrads!(params)
    backward!(loss)
    NN.optimize!(optimizer)
    if (iter % 100 == 0)
        @info iter
        val_x = reshape(collect(0:0.01:1), :, 1)
        val_pred = NN.apply(model, Tensor(val_x), params).values
        p1 = scatter(x.values, y.values, label="data")
        plot!(p1, 0:0.01:1, val_pred, title=(@sprintf "[iter %4i loss : %.6f]" iter loss.values), label="predict")
        p2 = plot(history, title="learning curve", xlims=(0, n_iter))
        plt = plot(p1, p2, size=(800, 1000), layout=(2, 1))
        frame(anim, plt)
    end
end


gif(anim, fps=10, "learning.gif")

