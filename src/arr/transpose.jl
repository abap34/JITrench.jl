import Base

mutable struct Transpose <: Functional
    grad_field::GradField
end

function forward(::Transpose, x)
    return transpose(x)
end

function backward(::Transpose, gys)
    return transpose(gys)
end

Base.transpose(x::Variable) = Transpose(GradField())(x)