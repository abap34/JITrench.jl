abstract type Device end

struct CPU <: Device end

struct GPU <: Device
    idx :: Int64
    function GPU(idx::Int64)
        if idx < 0
            throw()
        end
    return new(idx)
    end
end
