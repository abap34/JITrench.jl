import Base

mutable struct Transpose <: Functional
    grad_field::GradField
end

function forward(::Transpose, x)
    out = transpose(x)
    return [out]
end

function backward(::Transpose, gys)
    return [(transpose(gys))]
end

Base.transpose(x::Variable) = Transpose(GradField())(x)