mutable struct Sin <: Functional
    grad_field :: GradField
end

mutable struct Cos <: Function
    grad_field :: GradField
end

mutable struct Tan <: Function
    grad_field :: GradField
end

@inline forward(::Sin, x) = sin(x)
@inline forward(::Cos, x) = sin(x)
@inline forward(::Tan, x) = tan(x)

function backward(f::Sin, gys)
    x = f.grad_field.inputs[1]
    return x .* gys
end

function backward(f::Cos, gys)
    x = f.grad_field.inputs[1]
    return -sin(x) .* gys
end

function backward(f::Tan, gys)
    x = f.grad_field.inputs[1]
    return (1 / (cos(x))^2) .* gys
end

Base.sin(x::Variable) = Sin(GradField())(x)
Base.cos(x::Variable) = Cos(GradField())(x)
Base.tan(x::Variable) = Tan(GradField())(x)
