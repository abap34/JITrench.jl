import Base
using Random

struct DataLoader
    dataset :: Dataset
    batch_size :: Int
    shuffle :: Bool
    index :: Vector{Int}
    function DataLoader(dataset; batch_size=1, shuffle=false)
        new(dataset, batch_size, shuffle, zeros(Int, length(dataset)))
    end
end

function Base.iterate(loader::DataLoader)
    if loader.batch_size > length(loader.dataset)
        # TODO: better error
        throw(DomainError("batch size > data length error"))
    end
    loader.index .= randperm(length(loader.dataset))
    data = loader.dataset[1:loader.batch_size]
    head = loader.batch_size + 1
    return (data, head)
end

function Base.iterate(loader::DataLoader, head::Int)
    if head == -1
        return nothing
    end
    if head + loader.batch_size > length(loader.dataset)
        head = length(loader.dataset) - loader.batch_size
        data = loader.dataset[head:head + loader.batch_size]
        head = head + loader.batch_size + 1
        return (data, -1)
    end
    if head > length(loader.dataset)
        loader.index = randperm(length(dataset))
        return nothing
    end 
    data = loader.dataset[head:head + loader.batch_size]
    head = head + loader.batch_size + 1
    return (data, head)
end
