mutable struct SumTo <: DiffableFunction
    grad_field::GradField
    shape
    x_shape
    SumTo(shape, x_shape) = new(GradField(), shape, x_shape)
end



function _sum_to(x, shape)
    if shape == ()
        is_scalar = true
        shape = (1,)
    end
    x_shape = size(x)
    (shape == x_shape) && (return x)
    sum_axis = findall(x -> x != 0, shape .- x_shape) 
    result = _sum(x, dims=sum_axis)
    if shape == (1,) && is_scalar
        if prod(size(result)) == 1
            return result[1]
        else
            throw(DimensionMismatch("fault sum_to. result: $result, result must be scalar."))
        end
    end
    if size(result) != shape
        throw(DimensionMismatch("fault sum_to. result: $result, shape:$shape"))
    else
        return result
    end
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

