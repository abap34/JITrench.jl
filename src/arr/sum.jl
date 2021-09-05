# this will be overloaded in ./broadcast.jl
function broadcast_to end

mutable struct Sum <: DiffableFunction
    grad_field::GradField
    dims
    keepdims
    x_shape
    Sum(dims, keepdims) = new(GradField(), dims, keepdims, nothing)
end


function set_same_dim(x::Number, n_dim)
    (ndims(x) > n_dim) && (throw(DimensionMismatch("ndims(x) = $(ndims(x)), but n_dim = $(n_dim). ndims(x) must be less than n_dim")))
    return reshape([x], (ones(Int, n_dim)...))
end



function set_same_dim(x::AbstractArray, n_dim)
    (ndims(x) > n_dim) && (throw(DimensionMismatch("ndims(x) = $(ndims(x)), but n_dim = $(n_dim). ndims(x) must be less than n_dim")))
    return reshape(x, (size(x)..., ones(Int, n_dim - ndims(x))...))
end




# wrapper of Base.sum: this function enabales `Base.sum` to use dims, keepdims
function _sum(x; dims=nothing, keepdims=false)
    if dims isa Nothing
        result = Base.sum(x)
        if keepdims 
            result = set_same_dim(result, ndims(x))
        end
    else
        result = Base.sum(x, dims=dims)
    end
    return result
end

function forward(f::Sum, x)
    f.x_shape = size(x)
    return _sum(x, dims=f.dims, keepdims=f.keepdims)
end

function backward(f::Sum, gy)
    return broadcast_to(gy, f.x_shape)
end


"""
    sum(x::Variable; dims=nothing, keepdims=false)

# Arguments
- keepdims
When true, the number of dimensions in the input and output arrays is guaranteed to match.

# Examples
```julia-repl
julia> x = Variable([1, 2, 3])
name: nothing 
values: [1, 2, 3]
creator: User-Defined(nothing)

julia> y = sum(x)
name: nothing 
values: 6
creator: JITrench.Sum

julia> x = Variable(rand(2, 2, 2))
name: nothing 
values: [0.8036752887616154 0.918293421092931; 0.9092024610524445 0.8788506330169976]

[0.7726260228472543 0.6884237844746439; 0.38214845033860456 0.660455509851009]
creator: User-Defined(nothing)

julia> y = sum(x, keepdims=true)
name: nothing 
values: [6.013675571435501]
creator: JITrench.Sum

julia> y.values
1×1×1 Array{Float64, 3}:
[:, :, 1] =
 6.013675571435501
```
"""
Base.sum(x::Variable; dims=nothing, keepdims=false) = Sum(dims, keepdims)(x)