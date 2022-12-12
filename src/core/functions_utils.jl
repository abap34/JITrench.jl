using .AutoDiff

get_values(x::T) where T <: Variable = x.values
get_values(x) = x



ones_like(x::R) where {R <: Real} = one(R)

ones_like(x::AbstractArray{R}) where {R <: Real} = ones(R, size(x))



@inline function cleargrad!(x::T) where T <: Variable
    x.grad = nothing
end

function as_tuple(x)
    return (x, )
end

function as_tuple(x::Tuple)
    return x
end
