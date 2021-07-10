mutable struct _Linear <: DiffableFunction
    grad_field :: GradField
end

function forward(::_Linear, x, W, b)
    return x * W .+ b
end

function backward(f::_Linear, gy)
    x, W, b = f.grad_field.inputs
    gx = matmul(gy, transpose(W))
    gW = matmul(transpose(x), gy)
    gb = sum_to(gy, size(b.values)) 
    return gx, gW, gb
end

linear(x, W, b) = _Linear(GradField())(x, W, b)
