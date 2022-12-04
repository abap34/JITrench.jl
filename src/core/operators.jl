using .AutoDiff
import .AutoDiff: AutoDiff.forward

import Base

mutable struct Add <: BinaryOperation
    grad_field::GradField
end

mutable struct Sub <: BinaryOperation
    grad_field::GradField
end

mutable struct Neg <: BinaryOperation
    grad_field::GradField
end

mutable struct Mul <: BinaryOperation
    grad_field::GradField
end


mutable struct Div <: BinaryOperation
    grad_field::GradField
end


mutable struct Pow{T} <: BinaryOperation
    grad_field::GradField
    c::T
end


@inline AutoDiff.forward(::Add, x1, x2) = x1 + x2
# case: a - b
@inline AutoDiff.forward(::Sub, x1, x2) = x1 - x2
# case: -a
@inline AutoDiff.forward(::Sub, x) = -x
@inline AutoDiff.forward(::Neg, x) = -x
@inline AutoDiff.forward(::Mul, x1, x2) = x1 * x2
@inline AutoDiff.forward(::Div, x1, x2) = x1 / x2
@inline AutoDiff.forward(f::Pow, x1) = x1^f.c


function backward(::Add, gy::Scalar)
    return (gy, gy)
end

function backward(::Add, gy::T) where T <: AbstractTensor 
    @. return (gy, gy)
end

function backward(::Sub, gy::Scalar)
    return (gy, -gy)
end

function backward(::Sub, gy::T) where T <: AbstractTensor
    return (gy, -gy)
end

function backward(::Neg, gy::Scalar)
    return -gy
end

function backward(::Neg, gy::T) where T <: AbstractTensor 
    return -gy
end

function backward(f::Mul, gy::Scalar)
    x1, x2 = f.grad_field.inputs
    return (x2 * gy, x1 * gy)
end

function backward(f::Mul, gy::T) where T <: AbstractTensor
    x1, x2 = f.grad_field.inputs
    @. return (x2 .* gy, x1 .* gy)
end

function backward(f::Div, gy::Scalar) r
    x1, x2 = f.grad_field.inputs
    return (1 / x2) * gy, (-x1 / (x2 * x2)) * gy
end

function backward(f::Div, gy::T) where T <: AbstractTensor
    x1, x2 = f.grad_field.inputs
    @. return (1 / x2) * gy, (-x1 / (x2 * x2)) * gy
end    


function backward(f::Pow, gy::Scalar) 
    x = f.grad_field.inputs[1]
    c = f.c
    return (c * (x^(c - 1))) * gy
end


function backward(f::Pow, gy::T) where T <: AbstractTensor
    x = f.grad_field.inputs[1]
    c = f.c
    @. return (c * (x^(c - 1))) * gy
end

const normal_operators = Dict(
    :+ => Add,
    :- => Sub,
    :* => Mul,
    :/ => Div,
)

Base.:-(x::T) where T <: Variable = Neg(GradField())(x)
pure_func(::Neg) = -
Base.:^(x::Variable, c) = Pow(GradField(), c)(x)

is_support(::typeof(^)) = true
jt_func(::typeof(^)) = Pow
pure_func(::Pow) = ^

for (op, jt_func) in normal_operators
    @eval is_support(::typeof(Base.$op)) = true
    @eval is_support_broadcast(::typeof(Base.$op)) = true
    @eval jt_func(::typeof(Base.$op)) = $jt_func
    @eval pure_func(::$(jt_func)) = Base.$op
    @eval Base.$op(x1::T, x2::R) where {T <: Variable, R <: Real} = Base.$op(promote(x1, x2)...)
    @eval Base.$op(x1::R, x2::T) where {T <: Variable, R <: Real} = Base.$op(promote(x1, x2)...)
    @eval Base.$op(x1::T, x2::S) where {T <: Variable, S <: Variable} = $jt_func(GradField())(x1, x2)
end
