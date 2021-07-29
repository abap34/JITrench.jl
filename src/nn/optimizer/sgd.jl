mutable struct SGD <: Optimizer
    lr
    target 
    SGD(;lr=1e-3) = new(lr, nothing)
end

function do_optimize!(sgd::SGD)
    for p in parameters(sgd.target)
        p.values -= sgd.lr * p.grad.values
    end
end

function setup!(sgd::SGD, model::Model)
    sgd.target = model
end
