import Base
using .AutoDiff
import .AutoDiff: forward, backward, call!

struct ReshapeField{T <: Tuple, S <: Tuple} <: AdditionalField
    in_shape :: T
    out_shape :: S
end


struct Reshape{T, S} <: UnaryOperator
    grad_field :: GradField
    additional_field :: ReshapeField{T, S}
end


function forward(::Type{Reshape}, reshape_field::ReshapeField, x)
    return reshape(x, reshape_field.out_shape)
end

function backward(f::Reshape, gy)
    in_shape = f.additional_field.in_shape
    return reshape(gy, in_shape)
end

function Base.reshape(x::T, out_shape) where T <: AbstractTensor
    in_shape = size(x)
    return call!(Reshape, ReshapeField(in_shape, out_shape), x)
end