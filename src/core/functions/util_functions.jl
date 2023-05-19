using .AutoDiff
import .AutoDiff: forward, backward, call!
import Base

struct ClampField{S <: Number, T <: Number} <: AdditionalField
    lo :: S
    hi :: T
end

struct Clamp{S, T} <: UnaryOperator
    grad_field :: GradField
    additional_field :: ClampField{S, T}
end

function forward(::Type{Clamp}, clamp_field::ClampField, x)
    return Base.clamp(x, clamp_field.lo, clamp_field.hi)
end

function backward(f::Clamp, gy::ScalarTypes)
    x = f.grad_field.inputs[1]
    return Int((x.values >= f.additional_field.lo) && (x.values <= f.additional_field.hi)) * gy
end

function backward(f::Clamp, gy::TensorTypes)
    x = f.grad_field.inputs[1]
    @. return Int((x.values >= f.additional_field.lo) && (x.value <= f.additional_field.hi)) * gy
end

Base.clamp(x::Scalar, lo, hi) = call!(Clamp, ClampField(lo, hi), x)

function Base.clamp(x::AbstractTensor, lo, hi) 
    call!(Clamp, ClampField(lo, hi), x)
end
