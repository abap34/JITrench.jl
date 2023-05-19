import Base

struct Dataset{T1 <: AbstractTensor, T2 <: AbstractTensor}
    X :: T1
    y :: T2
end

function Base.length(dataset::Dataset)
    return length(dataset.y)
end

function Base.getindex(dataset::Dataset, index)
    return dataset.X[index, :], dataset.y[index]
end

