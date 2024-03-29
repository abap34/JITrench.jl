struct FlattenField{T <: Tuple} <: AdditionalField
    in_shape::T
end


struct Flatten{T} <: UnaryOperator
    grad_field::GradField
    additional_field::FlattenField{T}
end


function forward(::Type{Flatten}, additional_field, x)
    return vcat(x...)
end

function backward(f::Flatten, gy)
    in_shape = f.additional_field.in_shape
    return reshape(gy, in_shape)
end


flatten(x::T) where {T <: AbstractTensor} = call!(Flatten, FlattenField(size(x)), x)
