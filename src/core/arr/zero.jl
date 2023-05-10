struct Zero <: UnaryOperator
    grad_field::GradField
end

forward(::Type{Zero}, x) = zero(x)

function backward(f::Zero, gy::Union{ScalarTypes, TensorTypes})
    x = f.grad_field.inputs[1]
    return zero(x)
end


zero(x::Variable) = call!(Zero, x)
    