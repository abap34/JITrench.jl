mutable struct Sigmoid <: DiffableFunction
    grad_field :: GradField
end

@inline forward(::Sigmoid, x) = _sigmoid(x)

function backward(f::Sigmoid, gy)
    y = f.grad_field.outputs[1]
    @. return (y * (1 - y)) * gy
end

function _sigmoid(x)
    1 / (1 + exp(-x))
end

sigmoid(x::Variable) = Sigmoid(GradField())(x)

get_jt_struct(::typeof(sigmoid)) = Sigmoid

Base.broadcasted(::typeof(sigmoid), x::Variable)  = Broadcasting(_sigmoid, Sigmoid(GradField()))(x)

