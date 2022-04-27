mutable struct Sigmoid <: SingleReturnFunction
    grad_field :: GradField
end


function _sigmoid(x::T) where {T <: Real}
    one(T) / (one(T) + exp(-x))
end

@inline forward(::Sigmoid, x) = _sigmoid(x)

function backward(f::Sigmoid, gy::Variable{T}) where {T <: Real}
    y = f.grad_field.outputs[1]
    return (y * (1 - y)) * gy
end

function backward(f::Sigmoid, gy::Variable{T}) where {T <: AbstractArray}
    y = f.grad_field.outputs[1]
    @. return (y * (1 - y)) * gy
end

sigmoid(x::Variable) = Sigmoid(GradField())(x)

Base.broadcasted(::typeof(sigmoid), x::Variable)  = Broadcasting(_sigmoid, Sigmoid(GradField()))(x)








