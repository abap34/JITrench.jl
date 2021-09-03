mutable struct GetIndex <: DiffableFunction
    ind
    grad_field :: GradField
    GetIndex(ind) = new(ind, GradField())
end

function forward(f::GetIndex, x)
    return x[f.ind...]
end

function backward(f::GetIndex, gy)
    x = f.grad_field.inputs[1]
    return GetIndexGrad(f.ind, size(x))(gy)
end

mutable struct GetIndexGrad <: DiffableFunction
    ind
    in_shape
    grad_field :: GradField
    GetIndexGrad(ind, in_shape) = new(ind, in_shape, GradField())
end

function add_at(arr::Vector, ind, val) 
    arr[ind...] += val
    return arr
end


function add_at(arr::AbstractArray, ind, val) 
    arr[ind...] .+= val
    return arr
end

function forward(f::GetIndexGrad, gy)
    gx = zeros(f.in_shape)
    return add_at(gx, f.ind, gy)
end

function backward(::GetIndexGrad, ggx)
    return GetIndex(ind, size(ggx))(ggx)
end

"""
    Base.getindex(x::Variable, ind...)
return `x.values[ind...]` as `Variable`.

# Examples
```julia-repl
julia> x = Variable(rand(2, 2))
name: nothing 
values: [0.7050007249265509 0.5075375401538957; 0.9953109600473362 0.8447135817368259]
creator: User-Defined(nothing)

julia> x[1, 2]
name: nothing 
values: 0.5075375401538957
creator: JITrench.GetIndex

julia> x[1, :]
name: nothing 
values: [0.7050007249265509, 0.5075375401538957]
creator: JITrench.GetIndex
```
"""
Base.getindex(x::Variable, ind...) = GetIndex(ind)(x)

is_support(::typeof(Base.getindex)) = true
get_jt_struct(::typeof(Base.getindex)) = GetIndex