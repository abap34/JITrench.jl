using DataStructures

"""
    get_values(x)

Get values from `x`. If `x` is `Variable`, return `x.values`. Otherwise, return `x`.
"""
@inline get_values(x::T) where {T <: Variable} = x.values
@inline get_values(x) = x

@inline ones_like(::R) where {R <: Real} = one(R)

@inline ones_like(x::AbstractArray{R}) where {R <: Real} = ones(R, size(x))

@inline ones_like(::Scalar{R}) where {R <: Real} = Scalar(one(R))

@inline ones_like(::Tensor) = Tensor(ones(eltype(x.values), size(x.values)))

@inline function cleargrad!(x::Variable)
    x.grad = nothing
end

@inline function as_tuple(x)
    return (x,)
end

@inline function as_tuple(x::Tuple)
    return x
end
