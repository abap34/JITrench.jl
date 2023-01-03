import LinearAlgebra

struct MatMul <: BinaryOperator
    grad_field :: GradField
end


function forward(::Type{MatMul}, x, W)
    return x * W
end


function backward(f::MatMul, gy)
    x, W = f.grad_field.inputs
    gx = matmul(gy, transpose(W))
    gW = matmul(transpose(x), gy)
    return gx, gW
end

matmul(x, W) = call!(MatMul, x, W)

Base.:*(x::T, W::S) where {T <: AbstractTensor, S <: AbstractTensor} = matmul(x, W)
