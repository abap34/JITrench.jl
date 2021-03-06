mutable struct Sin <: SingleReturnFunction
    grad_field::GradField
end

mutable struct Cos <: SingleReturnFunction
    grad_field::GradField
end

mutable struct Tan <: SingleReturnFunction
    grad_field::GradField
end

mutable struct Log <: SingleReturnFunction
    grad_field::GradField
end

mutable struct Exp <: SingleReturnFunction
    grad_field::GradField
end

@inline forward(::Sin, x) = sin(x)
@inline forward(::Cos, x) = cos(x)
@inline forward(::Tan, x) = tan(x)
@inline forward(::Log, x) = log(x)
@inline forward(::Exp, x) = exp(x)


function backward(f::Sin, gy::Variable{T}) where {T<:Real}
    x = f.grad_field.inputs[1]
    return cos(x) * gy
end


function backward(f::Sin, gy::Variable{T}) where {T<:AbstractArray}
    x = f.grad_field.inputs[1]
    @. return cos(x) * gy
end




function backward(f::Cos, gy::Variable{T}) where {T<:Real}
    x = f.grad_field.inputs[1]
    return -sin(x) * gy
end

function backward(f::Cos, gy::Variable{T}) where {T<:AbstractArray}
    x = f.grad_field.inputs[1]
    @. return -sin(x) * gy
end





function backward(f::Tan, gy::Variable{T}) where {T<:Real}
    x = f.grad_field.inputs[1]
    return (1 / (cos(x)^2)) * gy
end

function backward(f::Tan, gy::Variable{T}) where {T<:AbstractArray}
    x = f.grad_field.inputs[1]
    @. return (1 / (cos(x)^2)) * gy
end




function backward(f::Log, gy::Variable{T}) where {T<:Real}
    x = f.grad_field.inputs[1]
    return gy / x
end


function backward(f::Log, gy::Variable{T}) where {T<:AbstractArray}
    x = f.grad_field.inputs[1]
    @. return gy / x
end





function backward(f::Exp, gy::Variable{T}) where {T<:Real}
    x = f.grad_field.inputs[1]
    return exp(x) * gy
end


function backward(f::Exp, gy::Variable{T}) where {T<:AbstractArray}
    x = f.grad_field.inputs[1]
    @. return exp(x) * gy
end


const math_functions = Dict(
    :(Base.sin) => Sin,
    :(Base.cos) => Cos,
    :(Base.tan) => Tan,
    :(Base.log) => Log,
    :(Base.exp) => Exp
)

for (func, jt_func) in math_functions
    @eval $func(x::Variable) = $jt_func(GradField())(x)
    @eval is_support(::typeof($func)) = true
    @eval is_support_broadcast(::typeof($func)) = true
    @eval jt_func(::typeof($func)) = $jt_func
    @eval pure_func(::$(jt_func)) = $func
end

