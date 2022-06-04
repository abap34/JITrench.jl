mutable struct SGD{F} <: Optimizer
    target_layers::F
    lr
end


SGD(target_layers; lr=1e-3) = SGD(target_layers, lr)

function optimize!(model::T, optimizer::SGD) where {T<:Model}
    for layer in optimizer.target_layers(model)
        for p in parameters(layer)
            p.values -= optimizer.lr * p.grad.values
        end
    end
end

