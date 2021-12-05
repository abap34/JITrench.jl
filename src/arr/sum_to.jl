mutable struct SumTo <: DiffableFunction
    grad_field::GradField
    shape
    x_shape
    SumTo(shape, x_shape) = new(GradField(), shape, x_shape)
end


function _sum_to(x, shape::Tuple{}) 
    return sum(x)
end


function _sum_to(x, shape) 
    x_shape = size(x)
    (shape == x_shape) && (return x)
    if length(x_shape) < length(shape)
        throw(DimensionMismatch("Unexpected shape. length(x_shape) must be bigger than length(shape). But length(x_shape) = $(length(x_shape)), length(shape) = $(length(shape))"))
    elseif length(x_shape) > length(shape)
        n_repeat = length(x_shape) - length(shape)
        repeated_axis = vcat([shape...], ones(Int, n_repeat))
        shape = (repeated_axis...,) 
        sum_axis = findall(x -> x != 0, shape .- x_shape) 
        result = _sum(x, dims=sum_axis)
        drop_dim = ((length(x_shape) - n_repeat + 1:length(x_shape))...,)
        result = dropdims(result, dims=drop_dim)
    else
        sum_axis = findall(x -> x != 0, shape .- x_shape) 
        result = _sum(x, dims=sum_axis)
    end
    return result
end

function forward(f::SumTo, x)
    f.x_shape = size(x)
    y = _sum_to(x, f.shape)
    return y 
end

function backward(f::SumTo, gy)
    return broadcast_to(gy, f.x_shape)
end


"""
    sum_to(x, shape)
Take the sum of each axis to form a `shape`.

# Examples
```julia-repl
julia> x = Variable(rand(2, 3))
name: nothing 
values: [0.2397911359535343 0.34270903251201146 0.699060178623987; 0.2345132451371843 0.21845435948625758 0.2924942369518322]
creator: User-Defined(nothing)

julia> JITrench.sum_to(x, (2, 1))
name: nothing 
values: [1.2815603470895327; 0.7454618415752741]
creator: JITrench.SumTo
```
"""
sum_to(x, shape) = SumTo(shape, size(x.values))(x)

