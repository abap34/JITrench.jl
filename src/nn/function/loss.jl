struct MeanSquaredError <: BinaryOperator
    grad_field :: GradField
end

forward(::Type{MeanSquaredError}, x1, x2) = sum((x1 - x2).^2) / length(x1)

function backward(f::MeanSquaredError, gy::ScalarTypes)
    x1, x2 = f.grad_field.inputs
    diff = x1 - x2
    gx1 = (fill(2gy, size(diff)) .* diff) / length(diff)
    return gx1, -gx1
end

mean_squared_error(x1::Tensor, x2::Tensor) = call!(MeanSquaredError, x1, x2)
