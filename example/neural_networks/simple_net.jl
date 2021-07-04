using JITrench 
using Random
using Printf
using Plots


Random.seed!(10)

function generate_datset(N)
    x = rand(N, 1) 
    y =  sin.(2π .* x) .+ (rand(N, 1) .* 0.5)
    return x,  y
end

function linear(x, W, b)
    return  JITrench.matmul(x, W) .+ b
end

function predict(x, W1, b1, W2, b2)
    y = linear(x, W1, b1)
    y = JITrench.sigmoid.(y)
    y = linear(y, W2, b2)
    return y
end

function mse(y_true, y_pred)
    diff = y_true .- y_pred
    return JITrench.sum(diff .^ 2) / length(diff.values)
end

function train(x, y, W1_init, b1_init, W2_init, b2_init, n_iter; lr=1e-1, log_interval=1000)
    x = Variable(x)
    y = Variable(y)
    W1 = Variable(W1_init)
    b1 = Variable(b1_init)
    W2 = Variable(W2_init)
    b2 = Variable(b2_init)
    history = []
    for iter in 1:n_iter
        y_pred = predict(x, W1, b1, W2, b2)
        loss = mse(y, y_pred)
        JITrench.cleargrad!(W1)
        JITrench.cleargrad!(b1)
        JITrench.cleargrad!(W2)
        JITrench.cleargrad!(b2)
        backward!(loss)
        W1.values -= W1.grad.values * lr
        b1.values -= b1.grad.values * lr
        W2.values -= W2.grad.values * lr
        b2.values -= b2.grad.values * lr
        push!(history, loss.values)
        if (iter - 1) % log_interval == 0
            @printf "iters %4i [loss] %.2f\n" iter loss.values 
        end
    end
    return W1, b1, W2, b2, history
end

x, y = generate_datset(100)

input_dim = 1
hidden_dim = 10
out_dim = 1

W1_init = 0.01 .* rand(input_dim, hidden_dim)
b1_init = zeros(1, hidden_dim)
W2_init = 0.01 .* rand(hidden_dim, out_dim)
b2_init = zeros(1, out_dim)
n_iters = 20000

W1_trained, b1_trained, W2_trained, b2_trained, history = train(x, y, W1_init, b1_init, W2_init, b2_init, n_iters)

println("finish train.")

plt = scatter(x, y, label="data")
x = reshape(collect(0:0.01:1), :, 1)
y_pred = predict(Variable(x), W1_trained, b1_trained, W2_trained, b2_trained).values
plot!(plt, 0:0.01:1, y_pred, label="predict")
savefig("predict.png")

plot(history, title="learning curve", yaxis=:log10)
savefig("history.png")