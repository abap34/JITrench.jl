import Base

mutable struct Add <: SingleReturnFunction
    grad_field::GradField
end

mutable struct Sub <: SingleReturnFunction
    grad_field::GradField
end

mutable struct Neg <: SingleReturnFunction
    grad_field::GradField
end

mutable struct Mul <: SingleReturnFunction
    grad_field::GradField
end


mutable struct Div <: SingleReturnFunction
    grad_field::GradField
end


mutable struct Pow{T} <: SingleReturnFunction
    grad_field::GradField
    c::T
end


@inline forward(::Add, x1, x2) = x1 + x2
# case: a - b
@inline forward(::Sub, x1, x2) = x1 - x2
# case: -a
@inline forward(::Sub, x) = -x
@inline forward(::Neg, x) = -x
@inline forward(::Mul, x1, x2) = x1 * x2
@inline forward(::Div, x1, x2) = x1 / x2
@inline forward(f::Pow, x1) = x1^f.c


function backward(::Add, gy::Variable{T}) where {T<:Real}
    return (gy, gy)
end

function backward(::Add, gy::Variable{T}) where {T<:AbstractArray}
    @. return (gy, gy)
end

function backward(::Sub, gy::Variable{T}) where {T<:Real}
    return (gy, -gy)
end


function backward(::Sub, gy::Variable{T}) where {T<:AbstractArray}
    return (gy, -gy)
end

function backward(::Neg, gy::Variable{T}) where {T<:Real}
    return -gy
end

function backward(::Neg, gy::Variable{T}) where {T<:AbstractArray}
    @. return -gy
end

function backward(f::Mul, gy::Variable{T}) where {T<:Real}
    x1, x2 = f.grad_field.inputs
    return (x2 * gy, x1 * gy)
end

function backward(f::Mul, gy::Variable{T}) where {T<:AbstractArray}
    x1, x2 = f.grad_field.inputs
    @. return (x2 .* gy, x1 .* gy)
end

function backward(f::Div, gy::Variable{T}) where {T<:Real}
    x1, x2 = f.grad_field.inputs
    return (1 / x2) * gy, (-x1 / (x2 * x2)) * gy
end

function backward(f::Div, gy::Variable{T}) where {T<:AbstractArray}
    x1, x2 = f.grad_field.inputs
    @. return (1 / x2) * gy, (-x1 / (x2 * x2)) * gy
end


function backward(f::Pow, gy::Variable{T}) where {T<:Real}
    x = f.grad_field.inputs[1]
    c = f.c
    return (c * (x^(c - 1))) * gy
end


function backward(f::Pow, gy::Variable{T}) where {T<:AbstractArray}
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


Base.:-(x::Variable) = Neg(GradField())(x)
Base.:^(x::Variable, c) = Pow(GradField(), c)(x)
is_support(::typeof(^)) = true
base_to_jt(::typeof(^)) = Pow

for (op, jt_func) in normal_operators
    @eval is_support(::typeof(Base.$op)) = true
    @eval is_support_broadcast(::typeof(Base.$op)) = true
    @eval base_to_jt(::typeof(Base.$op)) = $jt_func
    @eval jt_to_base(::$(jt_func)) = Base.$op
    @eval Base.$op(x1::Variable, x2::Real) = Base.$op(promote(x1, x2)...)
    @eval Base.$op(x1::Real, x2::Variable) = Base.$op(promote(x1, x2)...)
    @eval Base.$op(x1::Variable, x2::Variable) = $jt_func(GradField())(x1, x2)
end
