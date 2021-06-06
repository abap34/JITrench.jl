import Base

mutable struct Add <: Functional
    grad_field::GradField
end

mutable struct Sub <: Functional
    grad_field::GradField 
end

mutable struct Neg <: Functional
    grad_field::GradField 
end

mutable struct Mul <: Functional
    grad_field::GradField 
end


mutable struct Div <: Functional
    grad_field::GradField 
end


mutable struct Pow <: Functional
    grad_field::GradField 
    c
end


@inline forward(::Add, x1, x2)  = x1 + x2
@inline forward(::Sub, x1, x2) = x1 - x2
@inline forward(::Neg, x) = -x
@inline forward(::Mul, x1, x2) = x1 * x2
@inline forward(::Div, x1, x2) = x1 / x2
@inline forward(f::Pow, x) = x^(f.c) 


function backward(::Add, gys)
    return (gys, gys)
end

function backward(::Sub, gys)
    return (gys, -gys)
end

function backward(::Neg, gys)
    return -gys
end

function backward(f::Mul, gys)
    x1, x2 = f.grad_field.inputs
    return (x2 * gys, x1 * gys) 
end

function backward(f::Div, gys)
    x1, x2 = f.grad_field.inputs
    return (1 / x2) * gys, (-x1 / (x2^2)) * gys 
end

function backward(f::Pow, gys)
    x, c = f.grad_field.inputs[1], f.c    
    return (c * (x^(c - 1))) * gys
end


add(x1::Variable, x2::Variable) = Add(GradField())(x1, x2)
sub(x1::Variable, x2::Variable) = Sub(GradField())(x1, x2)
neg(x::Variable) = Neg(GradField())(x)
mul(x1::Variable, x2::Variable) = Mul(GradField())(x1, x2)
div(x1::Variable, x2::Variable) = Div(GradField())(x1, x2)
pow(x1::Variable, x2::Variable) = Pow(GradField(), x2.values)(x1)


const operators = Dict(
    :+ => :add, 
    :- => :sub, 
    :* => :mul,
    :/ => :div,
    :^ => :pow
)

Base.:-(x::Variable) = neg(x)

for (op, jt_func) in operators
    @eval Base.$op(x1::Variable, x2::Real) = Base.$op(promote(x1, x2)...)
    @eval Base.$op(x1::Real, x2::Variable) = Base.$op(promote(x1, x2)...)
    @eval Base.$op(x1::Variable, x2::Variable) = $jt_func(x1, x2)
end

    


