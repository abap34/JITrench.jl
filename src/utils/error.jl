using .AutoDiff

struct NotSameShapeError <: Exception
    shapes
end

Base.showerror(io::IO, e::NotSameShapeError) = print(io,
    "Arguments must be same shape, Arguments shaps are $(shape).",
)

check_sameshape(args...) = allequal(size.(args))

struct NotSameLengthError <: Exception
    x_len
    y_len
end

Base.showerror(io::IO, e::NotSameLengthError) = print(io,
    "Arguments must be same length. Input length are ($(e.x_len), $(e.y_len)",
)

function check_samelength(x::AbstractArray, y::AbstractArray)
    x_len = length(x)
    y_len = length(y)
    return x_len != y_len
end

struct BroadcastCallError <: Exception end

Base.showerror(io::IO, e::BroadcastCallError) =
    print(io, "Please call this function with `req_broadcast`")

function check_broadcastable(x::T) where {T<:Variable}
    if !(x.req_broadcast)
        throw(BroadcastCallError())
    end
end

function check_broadcastable(args...)
    for arg in args
        if arg.req_broadcast
            return true
        end
    end
    return false
end