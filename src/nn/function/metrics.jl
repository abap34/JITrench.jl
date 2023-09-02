function accuracy(::AbstractArray{<:AbstractFloat}, ::AbstractArray{<:AbstractFloat})
    throw(DomainError("Accuracy is not defined for floating point arrays."))
end
    

function accuracy(x::AbstractArray, y::AbstractArray)
    x_len = length(x)
    y_len = length(y)
    return count(x .== y) / x_len
end



