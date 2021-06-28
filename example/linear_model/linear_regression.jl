using JITrench 
using Random
using Printf
using UnicodePlots



Random.seed!(10)

function generate_datset(N)
    x = rand(N) * 10
    y = 2x .+ 3 .+ (rand(N) .* 5)
    return x, y
end

function predict(x, W, b)
    y = JITrench.matmul(x, W) .+ b
    return y
end

function mse(y_true, y_pred)
    diff = y_true .- y_pred
    return JITrench.sum(diff .* diff) / length(diff.values)
end


function train(x, y, W_init, b_init, n_iter; lr=0.01)
    x = Variable(x)
    y = Variable(y)
    W = Variable(W_init)
    b = Variable(b_init)
    for iter in 1:n_iter
        y_pred = predict(x, W, b)
        loss = mse(y, y_pred)
        JITrench.cleargrad!(W)
        JITrench.cleargrad!(b)
        backward!(loss)
        W.values -= W.grad.values * lr
        b.values -= b.grad.values * lr
        @printf "iters %4i [loss] %.2f [W] %.2f [b] %.2f\n" iter loss.values W.values b.values
    end
    return W, b
end




x, y = generate_datset(100)
W = 1
b = 1
n_iters = 100

W_trained, b_trained = train(x, y, W, b, n_iters)


println("finish train.")
@show W_trained
@show b_trained

plt = scatterplot(x, y, color=:red, name="data")
lineplot!(plt, 1:0.1:10, predict(Variable(collect(1:0.1:10)), W_trained, b_trained).values, name="predict")

