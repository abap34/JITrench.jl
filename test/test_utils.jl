using Random

function numerical_diff(f::Function, x::Real; e=1e-7)
    return (f(x + e) - f(x - e)) / 2e
end

function numerical_diff(f::Function, xs::AbstractArray; e=1e-7)
    grads = zeros(length(xs)) 
    for idx in 1:length(xs)
        tmp_val = xs[idx]
        xs[idx] = tmp_val + e
        fxh1 = f(xs...)
        xs[idx] = tmp_val - e
        fxh2 = f(xs...)
        grads[idx] = (fxh1 - fxh2) / 2e
        xs[idx] = tmp_val
    end
    return grads
end

function backward_diff(f, x::R) where R <: Real
    x = Variable(x)
    y = f(x)
    backward!(y)
    return x.grad.values
end

function backward_diff(f, xs::AbstractArray)
    inputs = Variable.(xs)
    outs = f(inputs...)
    backward!(outs)
    return (input -> input.grad.values).(inputs)
end

function ≃(x, y; e=1e-4)
    return ((x - y) < 1e-15) || (-e < ((x - y) / y) < e)
end

function ≃(X::AbstractArray, Y::AbstractArray; e=1e-4)
    for (x, y) in zip(X, Y)
        if !(((x - y) < 1e-15) || (-e < ((x - y) / y) < e))
            return false
        end
    end
    return true
end


function randshape(N::Int; min_dim=1, max_dim=10)
    shape_length = rand(min_dim:max_dim) 
    facts = factor(Array, N)
    if shape_length > length(facts)
        append!(facts, ones(Int, shape_length - length(facts)))
    end
    assign = rand(1:shape_length, length(facts))
    return Tuple((i -> prod(getindex.(Ref(facts), findall(x -> x == i, assign)))).(1:shape_length))
end
