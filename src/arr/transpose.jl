import Base

mutable struct Transpose <: DiffableFunction
    in_is_vector
    grad_field::GradField
    Transpose(in_shape, grad_field) = new(length(in_shape) == 1, grad_field)
end

function forward(::Transpose, x)
    return transpose(x) 
end

function backward(f::Transpose, gy)
    gx = transpose(gy)
    if f.in_is_vector
        gx.values = gx.values[:]
    end
    return gx
end

Base.transpose(x::Variable) = Transpose(size(x.values), GradField())(x)