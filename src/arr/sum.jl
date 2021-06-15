# this will be overloaded in ./broadcast.jl
function broadcast_to(arg...) end

mutable struct Sum <: Functional
    grad_field::GradField
    dims
    keepdims
    x_shape
    Sum(dims, keepdims) = new(GradField(), dims, keepdims)
end


function set_same_dim(x, n_dim)
    (ndims(x) > n_dim) && (throw(DimensionMismatch("ndims(x) = $(ndims(x)), but n_dim = $(n_dim). ndims(x) must be less than n_dim")))
    if x isa Number
        return reshape([x], (ones(Int, n_dim)...))
    else
        return reshape(x, (size(x)..., ones(Int, n_dim - ndims(x))...))
    end
end


# wrapper of Base.sum: this function enabales `Base.sum` to use dims, keepdims
function _sum(x; dims=nothing, keepdims=false)
    if dims isa Nothing
        result = Base.sum(x)
        keepdims && (result = set_same_dim(result, ndims(x)))
    else
        result = Base.sum(x, dims=dims)
    end
    return result
end

function forward(f::Sum, x)
    f.x_shape = size(x)
    return [_sum(x, dims=f.dims, keepdims=f.keepdims)]
end

function backward(f::Sum, gy)
    return [broadcast_to(gy[1].values, f.x_shape)]
end

sum(x::Variable; dims=nothing, keepdims=false) = Sum(dims, keepdims)(x)