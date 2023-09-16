abstract type Device end

"""
    CPU <: Device

Repersents CPU device. This is used for the default device.
"""
struct CPU <: Device end

"""
    GPU(idx::Int64) <: Device

Repersents GPU device. `idx` is corresponding device index in CUDA.jl.
"""
struct GPU <: Device
    idx::Int64
    function GPU(idx::Int64)
        if idx < 0
            throw(ArgumentError("GPU index must be non-negative. Passed idx: $idx"))
        end
        return new(idx)
    end
end

"""
    NotSameDeviceError <: Exception

Exception thrown when the device of two tensors are not the same.
"""
struct NotSameDeviceError <: Exception
    same_accelerator::Bool
    same_gpu_idx::Bool
    function NotSameDeviceError(; same_accelerator, same_gpu_idx)
        if (same_accelerator) && (same_gpu_idx)
            throw(
                DomainError(
                    "same_accelerator and same_gpu_idx can never be false at the same time",
                ),
            )
        end
        return new(same_accelerator, same_gpu_idx)
    end
end


function Base.showerror(io::IO, e::NotSameDeviceError)
    if !(e.same_accelerator)
        print(
            io,
            "Arguments must be in the same device, Arguments are on both the CPU and the GPU.",
        )
    end

    if !(e.same_gpu_idx)
        print(
            io,
            "All arguments must be in the same device. Arguments are on different GPUs.",
        )
    end
end


"""
    check_same_device(device1::Device, device2::Device)

Check if `device1` and `device2` are the same device. 
If they are the same device, return nothing. Otherwise, throw `NotSameDeviceError`.

# Arguments
- `device1`: Device to be compared.
- `device2`: Device to be compared.

# Example
```julia-repl
julia> device1 = JITrench.CPU()
JITrench.AutoDiff.CPU()

julia> device2 = JITrench.GPU(0)
JITrench.AutoDiff.GPU(0)

julia> JITrench.check_same_device(device1, device2)
ERROR: All arguments must be in the same device. Arguments are on different GPUs.
```
"""
check_same_device(::T, ::T) where {T<:Device} = nothing

check_same_device(::CPU, ::GPU) = throw(NotSameDeviceError(same_accelerator=true, same_gpu_idx=false))

function check_same_device(device1::GPU, device2::GPU)
    if device1.idx != device2.idx
        throw(NotSameDeviceError(same_accelerator=true, same_gpu_idx=false))
    else
        return nothing
    end
end
