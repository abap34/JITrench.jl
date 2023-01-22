using .AutoDiff

struct NotSameShapeError <: Exception
    shapes
end

Base.showerror(io::IO, e::NotSameShapeError) = print(io, 
"Arguments must be same shape, Arguments shaps are $(shape).",
)

check_sameshape(args...) = allequal(size.(args)) 
struct NotImplemetedError <: Exception
    msg::String
end

Base.showerror(io::IO, e::NotImplemetedError) = print(io, e.msg)

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
    if !(same_accelerator)
        print(
            io,
            "Arguments must be in the same device, Arguments are on both the CPU and the GPU.",
        )
    end

    if !(same_gpu_idx)
        print(
            io,
            "All arguments must be in the same device. Arguments are on different GPUs.",
        )
    end
end


struct BroadcastCallError <: Exception end

Base.showerror(io::IO, e::BroadcastCallError) =
    print(io, "Please call this function with `req_broadcast`")

function check_broadcastable(x::T) where {T <: Variable}
    if !(x.req_broadcast)
        throw(BroadcastCallError())
    end
end
