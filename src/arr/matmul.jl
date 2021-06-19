import LinearAlgebra

mutable struct MatMul <: Functional
    grad_field :: GradField
end


function forward(::MatMul, x, W)
    return x * W
end


function backward(f::MatMul, gy)
    x, W = f.grad_field.inputs
    gx = matmul(gy, transpose(W))
    gW = matmul(transpose(x), gy)
    return gx, gW
end

matmul(x, W) = MatMul(GradField())(x, W)