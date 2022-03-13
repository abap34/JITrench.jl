using Primes

function randfact(N; min_len=1, max_len=10)
    l = rand(min_len:max_len)
    base_arr = factor(Array, N)
    ind = rand(1:l, length(base_arr))
    return prod.(getindex.(Ref(base_arr), (i -> ind .== i).(1:l)))
end


function generate_shape(n_elemtnt; min_dim=1, max_dim=5)
    return Tuple(randfact(n_elemtnt, max_len=max_dim)), Tuple(randfact(n_elemtnt, min_len=min_dim, max_len=max_dim))
end


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


function random_arr(;min_dim=1, max_dim=4, min_r=1, max_r=10, collection=Float64)
    dim = rand(min_dim:max_dim)
    shape = Tuple(rand(min_r:max_r, dim))
    return rand(collection, shape)
end