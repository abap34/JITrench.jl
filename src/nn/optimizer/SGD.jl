mutable struct SGD{F} <: AbstractOptimizer
    watching_param :: Parameter
    lr :: F
    function SGD(param, lr::F) where F
        new{F}(param, lr)
    end
end

function optimize!(sgd::SGD)
    for param in iterate_all(sgd.watching_param)
        param.values -= param.grad .* sgd.lr
    end
end
