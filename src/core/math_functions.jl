using .AutoDiff
import .AutoDiff: forward, call!
import Base

struct Sin <: UnaryOperator
    grad_field::GradField
end

struct Cos <: UnaryOperator
    grad_field::GradField
end

struct Tan <: UnaryOperator
    grad_field::GradField
end

struct Log <: UnaryOperator
    grad_field::GradField
end

struct Exp <: UnaryOperator
    grad_field::GradField
end

struct Square <: UnaryOperator
    grad_field::GradField
end

@inline forward(::Type{Sin}, x) = sin(x)
@inline forward(::Type{Cos}, x) = cos(x)
@inline forward(::Type{Tan}, x) = tan(x)
@inline forward(::Type{Log}, x) = log(x)
@inline forward(::Type{Exp}, x) = exp(x)
@inline forward(::Type{Square}, x) = x^2

function backward(f::Sin, gy::T) where T <: ScalarTypes
    x = f.grad_field.inputs[1]
    return cos(x) * gy
end

function backward(f::Cos, gy::T) where T <: ScalarTypes
    x = f.grad_field.inputs[1]
    return -sin(x) * gy
end

function backward(f::Tan, gy::T) where T <: ScalarTypes
    x = f.grad_field.inputs[1]
    return (1 / (cos(x)^2)) * gy
end

function backward(f::Log, gy::T) where T <: ScalarTypes
    x = f.grad_field.inputs[1]
    return gy / x
end


function backward(f::Exp, gy::T) where T <: ScalarTypes
    x = f.grad_field.inputs[1]
    return exp(x) * gy
end

function backward(f::Square, gy::T) where T <: ScalarTypes
    x = f.grad_field.inputs[1]
    return 2x * gy
end


Base.sin(x::T) where T <: Scalar = call!(Sin, x)
Base.cos(x::T) where T <: Scalar = call!(Cos, x)
Base.tan(x::T) where T <: Scalar = call!(Tan, x)
Base.log(x::T) where T <: Scalar = call!(Log, x)
Base.exp(x::T) where T <: Scalar = call!(Exp, x)

function Base.sin(x::T) where T <: AbstractTensor 
    check_broadcastable(x)
    call!(Sin, x)
end

function Base.cos(x::T) where T <: AbstractTensor 
    check_broadcastable(x)
    call!(Cos, x)
end

function Base.tan(x::T) where T <: AbstractTensor 
    check_broadcastable(x)
    call!(Tan, x)
end

function Base.log(x::T) where T <: AbstractTensor 
    check_broadcastable(x)
    call!(Log, x)
end

function Base.exp(x::T) where T <: AbstractTensor 
    check_broadcastable(x)
    call!(Exp, x)
end


Base.literal_pow(::typeof(^), x::T, ::Val{2}) where T <: Scalar = call!(Square, x)