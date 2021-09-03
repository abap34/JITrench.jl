import Base
mutable struct Transpose <: DiffableFunction
    in_is_vector
    grad_field::GradField
    Transpose(in_shape, grad_field) = new(length(in_shape) == 1, grad_field)
end

function forward(::Transpose, x)
    return transpose(x)
end

function backward(f::Transpose, gy)
    gx = transpose(gy)  
    (f.in_is_vector) && (return flatten(gx))
    return gx
end

"""
    transpose(x::Variable)
# Examples
```julia-repl
julia> x = Variable([1 2; 3 4])
name: nothing 
values: [1 2; 3 4]
creator: User-Defined(nothing)

julia> transpose(x)
name: nothing 
values: [1 3; 2 4]
creator: JITrench.Transpose
```
"""
Base.transpose(x::Variable) = Transpose(size(x.values), GradField())(x)