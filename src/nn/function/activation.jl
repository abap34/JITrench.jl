struct Sigmoid <: UnaryOperator
    grad_field::GradField
end

_sigmoid(x) = one(x) / (one(x) + exp(-x))

@inline forward(::Type{Sigmoid}, x) = _sigmoid(x)

function backward(f::Sigmoid, gy::ScalarTypes)
    y = f.grad_field.outputs[1]
    return (y * (1 - y)) * gy
end

function backward(f::Sigmoid, gy::TensorTypes) 
    y = f.grad_field.outputs[1]
    @. return (y * (1 - y)) * gy
end


sigmoid(x) = call!(Sigmoid, x)


struct ReLU <: UnaryOperator
    grad_field::GradField
end

_relu(x) = max(x, zero(x))

@inline forward(::Type{ReLU}, x) = _relu(x)

function backward(f::ReLU, gy::ScalarTypes)
    x = f.grad_field.inputs[1]
    if x < 0
        return 0
    else
        return 1
    end
end

function backward(f::ReLU, gy::TensorTypes)
    x = f.grad_field.inputs[1]

    mask = x .< 0
    return mask .* gy
end


relu(x) = call!(ReLU, x)
