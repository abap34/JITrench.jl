function accuracy(::AbstractArray{AbstractFloat}, ::AbstractArray{AbstractFloat})
    # TODO better impl
    throw(DomainError(""))
end
    

function accuracy(x::AbstractArray, y::AbstractArray)
    x_len = length(x)
    y_len = length(y)
    return count(x .== y) / x_len
end



