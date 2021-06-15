# TODO:better implementation
function _broadcast_to(A, shape)
    return zeros(shape) .+ A
end

mutable struct BroadcastTo
    grad_field::GradField
    shape
    x_shape
    BroadcastTo(shape, x_shape) = new(GradField(), shape, x_shape)
end

function forward(f::BroadcastTo, x)
    y = _broadcast_to(x, f.shape)
    return y
end

function backward(f::BroadcastTo, gy)
    gx = _sum_to(gy, f.x_shape)   
    return gx
end


function broadcast_to(x, shape)
    if size(x) == shape
        return Variable(shape)
    else
        BroadcastTo(shape, size(x))(x)
    end
end