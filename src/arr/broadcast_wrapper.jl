mutable struct BroadcastWrapper{F} <: DiffableFunction
    func :: F
end

Base.broadcasted(f, args::Variable...) = BroadcastWrapper(f)(args...)

function backward(f::BroadcastWrapper, gy)
    f_gradfield = f.grad_field
end

function req_broadcast!(vars)
    for var in vars
        var.req_broadcast = true
    end
end

function fin_broadcast!(vars)
    for var in vars
        var.req_broadcast = false
    end
end

function (f::BroadcastWrapper)(vars...) 
    req_broadcast!(vars)
    true_func = f.func
    y = f.func(vars...)
    fin_broadcast!(vars)
    return y
end
