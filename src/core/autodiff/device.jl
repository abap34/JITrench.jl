abstract type Device end

struct CPU <: Device end

struct GPU <: Device
    idx::Int64
    function GPU(idx::Int64)
        if idx < 0
            throw(ArgumentError("GPU index must be non-negative. Passed idx: $idx"))
        end
        return new(idx)
    end
end

check_same_device(device1::T, device2::T) where {T <: Device} = nothing

check_same_device(device1::CPU, device2::GPU) = throw(NotSameDeviceError(true, false))

function check_same_device(device1::GPU, device2::GPU)
    if device1.idx != device2.idx
        throw(NotSameDeviceError(same_accelerator = true, same_gpu_idx = false))
    else
        return device1.idx
    end
end
