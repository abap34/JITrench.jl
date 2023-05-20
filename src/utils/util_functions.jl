function logsumexp!(arr::AbstractArray, dims)
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


function one_hot_batch(labels::AbstractArray, nclass::Int)
    N = size(labels)[1]
    result = zeros(Bool, N, nclass)
    for (i, label) in enumerate(labels)
        result[i, Int(label)] = 1
    end
    return result
end

function one_hot_batch(labels::Tensor, nclass::Int)
    return Tensor(one_hot_batch(labels.values, nclass))
end