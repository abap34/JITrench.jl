mutable struct Flatten <: DiffableFunction
    in_shape
    grad_field :: GradField
    Flatten(in_shape, grad_field) = new(in_shape, grad_field)
end


function forward(::Flatten, x)
    return vcat(x...)
end

function backward(f::Flatten, gy)
    reshape(gy, f.in_shape)
end

"""
    flatten(x::Variable)
The function corresponding to `vcat(x...)`.

# Example
julia> x = Variable([1 2; 3 4; 5 6])
name: nothing 
values: [1 2; 3 4; 5 6]
creator: User-Defined(nothing)

julia> JITrench.flatten(x)
name: nothing 
values: [1, 3, 5, 2, 4, 6]
creator: JITrench.Flatten
"""
flatten(x::Variable) = Flatten(size(x.values), GradField())(x)
