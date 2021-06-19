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
end


@inline forward(::Add, x1, x2)  = x1 + x2
@inline forward(::Sub, x1, x2) = x1 - x2
@inline forward(::Neg, x) = -x
@inline forward(::Mul, x1, x2) = x1 * x2
@inline forward(::Div, x1, x2) = x1 / x2
@inline forward(f::Pow, x1, x2) = x1^x2 


function backward(::Add, gy)
    @. return (gy, gy)
end

function backward(::Sub, gy)
    @. return (gy, -gy)
end

function backward(::Neg, gy)
    @. return -gy
end

function backward(f::Mul, gy)
    x1, x2 = f.grad_field.inputs
    @. return (x2 .* gy, x1 .* gy) 
end

function backward(f::Div, gy)
    x1, x2 = f.grad_field.inputs
    @. return (1 / x2) * gy, (-x1 / (x2*x2)) * gy 
end

function backward(f::Pow, gy)
    x, c = f.grad_field.inputs  
    @. return (c * (x^(c - 1))) * gy, (x^c * (log(x))) * gy
end


add(x1::Variable, x2::Variable) = Add(GradField())(x1, x2)
sub(x1::Variable, x2::Variable) = Sub(GradField())(x1, x2)
neg(x::Variable) = Neg(GradField())(x)
mul(x1::Variable, x2::Variable) = Mul(GradField())(x1, x2)
div(x1::Variable, x2::Variable) = Div(GradField())(x1, x2)
pow(x1::Variable, x2::Variable) = Pow(GradField())(x1, x2)


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

    


