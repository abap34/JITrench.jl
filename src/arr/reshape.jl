import Base

mutable struct Reshape <: DiffableFunction
    grad_field::GradField
    in_shape :: Tuple
    out_shape :: Tuple
    Reshape(in_shape, out_shape) = new(GradField(), in_shape, out_shape)
end

function forward(f::Reshape, x)
    f.in_shape = size(x)
    return reshape(x, f.out_shape)
end

function backward(f::Reshape, gy)
    return reshape(gy, f.in_shape)
end

Base.reshape(x::Variable, shape) = Reshape(size(x), shape)(x)