mutable struct Sin <: Functional
    grad_field::GradField
end

mutable struct Cos <: Functional
    grad_field::GradField
end

mutable struct Tan <: Functional
    grad_field::GradField
end

mutable struct Log <: Functional
    grad_field::GradField
end



@inline forward(::Sin, x) = sin(x)
@inline forward(::Cos, x) = cos(x)
@inline forward(::Tan, x) = tan(x)
@inline forward(::Log, x) = log(x)

function backward(f::Sin, gy)
    x = f.grad_field.inputs[1]
    return cos(x) * gy
end

function backward(f::Cos, gy)
    x = f.grad_field.inputs[1]
    return -sin(x) * gy
end

function backward(f::Tan, gy)
    x = f.grad_field.inputs[1]
    return (1 / (cos(x)^2)) * gy
end


function backward(f::Log, gy)
    x = f.grad_field.inputs[1]
    return gy / x    
end

Base.sin(x::Variable) = Sin(GradField())(x)
Base.cos(x::Variable) = Cos(GradField())(x)
Base.tan(x::Variable) = Tan(GradField())(x)
Base.log(x::Variable) = Log(GradField())(x)