struct Transpose <: UnaryOperator
    grad_field::GradField
end

function forward(::Type{Transpose}, x)
    return transpose(x)
end

function backward(::Transpose, gy)
    return transpose(gy)
end

Base.transpose(x::T) where {T <: AbstractTensor} = call!(Transpose, x)
