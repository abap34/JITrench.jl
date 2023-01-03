struct SumField{T <: Tuple} <: AdditionalField
    in_shape :: T
end

struct Sum <: UnaryOperator
    grad_field::GradField
    additional_field :: SumField
end

function forward(::Type{Sum}, additional_field::SumField, x)
    return sum(x)
end

function backward(f::Sum, gy)
    in_shape = f.additional_field.in_shape
    return broadcast_to(gy, in_shape)
end

Base.sum(x::T) where T <: AbstractTensor = call!(Sum, SumField(size(x)), x)
