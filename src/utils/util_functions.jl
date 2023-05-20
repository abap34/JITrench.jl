function logsumexp!(arr::AbstractArray, dims=1)
    m = maximum(arr, dims=dims)
    arr .-= m
    arr .= exp.(arr)
    s = sum(arr, dims=dims)
    s .= log.(s)
    return s .+ m
end

function logsumexp!(arr::AbstractArray)
    m = maximum(arr)
    arr .-= m
    arr .= exp.(arr)
    return log(sum(arr)) + m
end