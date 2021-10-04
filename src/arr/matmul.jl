import LinearAlgebra

mutable struct MatMul <: DiffableFunction
    grad_field::GradField
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

"""
    matmul(x, W)
Matrix multiplication.

# Example

```julia-repl
julia> x = Variable([1 1; 0 1])
name: nothing 
values: [1 1; 0 1]
creator: User-Defined(nothing)

julia> W = Variable([1 0; 1 1])
name: nothing 
values: [1 0; 1 1]
creator: User-Defined(nothing)

julia> matmul(x, W)
name: nothing 
values: [2 1; 1 1]
creator: MatMul
```
"""
matmul(x, W) = MatMul(GradField())(x, W)

Base.:*(x::Variable{Matrix{T}}, W::Variable{Matrix{R}}) where {T <: Real, R <: Real} = matmul(x, W)
