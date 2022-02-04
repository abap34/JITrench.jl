get_values(x::Variable) = x.values
get_values(x) = x



ones_like(x::R) where {R <: Real} = one(R)

ones_like(x::AbstractArray{R}) where {R <: Real} = ones(R, size(x))


"""
    cleargrad!(x::Variable)
Reset the Variable's gradient.

```julia-repl
julia> x
name: nothing 
values: 1
grad: Variable(1)
creator: User-Defined(nothing)

julia> x.grad
name: nothing 
values: 1
creator: User-Defined(nothing)

julia> cleargrad!(x)

julia> x.grad === nothing
true
```
"""
@inline function cleargrad!(x::Variable)
    x.grad = nothing
end

function as_tuple(x)
    return (x,)
end

function as_tuple(x::Tuple)
    return x
end
