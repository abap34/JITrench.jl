function _broadcast_to(x::R, shape) where R <: Real
    return fill(x, shape)
end

function _broadcast_to(A::AbstractArray{R}, shape) where R <: Real
    return zeros(R, shape) .+ A
end

struct BroadcastToField{T <: Tuple, S <: Tuple} <: AdditionalField
    in_shape :: T
    out_shape :: S
end

struct BroadcastTo{T, S} <: UnaryOperator
    grad_field :: GradField
    additional_field :: BroadcastToField{T, S}    
end

function forward(::Type{BroadcastTo}, additional_field::BroadcastToField, x)
    shape = additional_field.out_shape
    y = _broadcast_to(x, shape)
    return y
end

function backward(f::BroadcastTo, gy)
    in_shape = f.additional_field.in_shape
    gx = sum_to(gy, in_shape)   
    return gx
end

function broadcast_to(x::T, shape) where T <: AbstractTensor
    if size(x.values) == shape
        return x
    else
        return call!(BroadcastTo, BroadcastToField(size(x), shape), x)
    end
end