struct Linear <: DiffableFunction
    grad_field :: GradField
end

forward(::Type{Linear}, x, W, b) = x * W .+ b

function backward(f::Linear, gy)
    x, W, b = f.grad_field.inputs
    gx = gy * transpose(W)
    gW = transpose(x) * gy
    gb = AutoDiff.sum_to(gy, size(b))
    return gx, gW, gb
end

linear(x, W, b) = call!(Linear, x, W, b)