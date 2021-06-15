mutable struct SumTo <: Functional
    grad_field::GradField
    shape
    x_shape
    SumTo(shape, x_shape) = new(GradField(), shape, x_shape)
end



function _sum_to(x, shape)
    x_shape = size(x)
    (shape == x_shape) && (return x)
    sum_axis = findall(x -> x != 0, shape .- x_shape) 
    result = _sum(x, dims=sum_axis)
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
    return [broadcast_to(gy, f.x_shape)]
end


sum_to(x, shape) = SumTo(shape, size(x.values))(x)

