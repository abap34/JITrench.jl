import Base

mutable struct Reshape <: Functional
    grad_field :: GradField
    in_shape
    out_shape
    Reshape(shape) = new(GradField(), nothing, shape)
end

function forward(f::Reshape, x)
    f.in_shape = size(x)
    return reshape(x, f.out_shape)
end

function backward(f::Reshape, gys)
    return reshape(gys, f.in_shape)
end

Base.reshape(x::Variable, shape) = Reshape(shape)(x)