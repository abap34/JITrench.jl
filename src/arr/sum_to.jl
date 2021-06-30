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
            return result
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


sum_to(x, shape) = SumTo(shape, size(x.values))(x)

