import Base

struct FillField{T <: Tuple} <: AdditionalField
    out_shape ::T
end

struct Fill{T} <: UnaryOperator
    grad_field::GradField
    additional_field::FillField{T}
end

function forward(::Type{Fill}, additional_field, x)
    out_shape = additional_field.out_shape
    return fill(x, out_shape)
end

function backward(f::Fill, gy)
    out_shape = f.additional_field.out_shape
    return out_shape .* gy
end

Base.fill(x::Scalar, v, shape) = call!(Fill, FillField(shape), v)