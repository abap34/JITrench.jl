import Base

mutable struct Add <: DiffableFunction
    grad_field::GradField
end

mutable struct Sub <: DiffableFunction
    grad_field::GradField 
end

mutable struct Neg <: DiffableFunction
    grad_field::GradField 
end

mutable struct Mul <: DiffableFunction
    grad_field::GradField 
end


mutable struct Div <: DiffableFunction
    grad_field::GradField 
end


mutable struct Pow <: DiffableFunction
    grad_field::GradField 
    c
end


@inline forward(::Add, x1, x2)  = x1 + x2
# case: a - b
@inline forward(::Sub, x1, x2) = x1 - x2
# case: -a
@inline forward(::Sub, x) = -x
@inline forward(::Neg, x) = -x
@inline forward(::Mul, x1, x2) = x1 * x2
@inline forward(::Div, x1, x2) = x1 / x2
@inline forward(f::Pow, x1) = x1^f.c


function backward(::Add, gy::Variable{T}) where {T <: Real}
    return (gy, gy)
end

function backward(::Add, gy::Variable{T}) where {T <: AbstractArray}
    @. return (gy, gy)
end

function backward(::Sub, gy::Variable{T}) where {T <: Real}
    return (gy, -gy)
end


function backward(::Sub, gy::Variable{T}) where {T <: AbstractArray}
    return (gy, -gy)
end

function backward(::Neg, gy::Variable{T})  where {T <: Real}
    return -gy
end

function backward(::Neg, gy::Variable{T})  where {T <: AbstractArray}
    @. return -gy
end

function backward(f::Mul, gy::Variable{T}) where {T <: Real}
    x1, x2 = f.grad_field.inputs
    return (x2 * gy, x1 * gy) 
end

function backward(f::Mul, gy::Variable{T}) where {T <: AbstractArray}
    x1, x2 = f.grad_field.inputs
    @. return (x2 .* gy, x1 .* gy) 
end

function backward(f::Div, gy::Variable{T})  where {T <: Real}
    x1, x2 = f.grad_field.inputs
    return (1 / x2) * gy, (-x1 / (x2 * x2)) * gy 
end

function backward(f::Div, gy::Variable{T})  where {T <: AbstractArray}
    x1, x2 = f.grad_field.inputs
    @. return (1 / x2) * gy, (-x1 / (x2 * x2)) * gy 
end


function backward(f::Pow, gy::Variable{T})  where {T <: Real}
    x = f.grad_field.inputs[1]
    c = f.c  
    return (c * (x^(c - 1))) * gy
end


function backward(f::Pow, gy::Variable{T})  where {T <: AbstractArray}
    x = f.grad_field.inputs[1]
    c = f.c  
    @. return (c * (x^(c - 1))) * gy
end

const normal_operators = Dict(
    :(Base.+) => Add,
    :(Base.-) => Sub, 
    :(Base.*) => Mul,
    :(Base./) => Div,
)


Base.:-(x::Variable) = Neg(GradField())(x)
Base.:^(x::Variable, c) = Pow(GradField(), c)(x)
is_support(::typeof(^)) = true
get_jt_struct(::typeof(^)) = Pow

for (op, jt_func) in normal_operators
    @eval is_support(::typeof($op)) = true
    @eval is_support_broadcast(::typeof($op)) = true
    @eval get_jt_struct(::typeof($op)) = $jt_func
    @eval $op(x1::Variable, x2::Real) = $op(promote(x1, x2)...)
    @eval $op(x1::Real, x2::Variable) = $op(promote(x1, x2)...)
    @eval $op(x1::Variable, x2::Variable) = $jt_func(GradField())(x1, x2)
end


