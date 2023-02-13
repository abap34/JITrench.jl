using CUDA
import Base


mutable struct Scalar{T <: Real} <: Variable
    values::T
    creator::Union{Nothing, DiffableFunction}
    grad::Union{Nothing, Scalar, Real}
    generation::Int
    name::Union{Nothing, String}
    req_broadcast::Bool
    device::CPU
    function Scalar(
        values::T;
        creator = nothing,
        grad = nothing,
        generation = 0,
        name = nothing,
        req_broadcast = false,
    ) where {T <: Real}
        new{T}(values, creator, grad, generation, name, req_broadcast, CPU())
    end
end

abstract type AbstractTensor <: Variable end

mutable struct Tensor{T <: AbstractArray} <: AbstractTensor
    values::T
    creator::Union{Nothing, DiffableFunction}
    grad::Union{Nothing, Tensor, AbstractArray}
    generation::Int
    name::Union{Nothing, String}
    req_broadcast::Bool
    device::CPU
    function Tensor(
        values::T;
        creator = nothing,
        grad = nothing,
        generation = 0,
        name = nothing,
        req_broadcast = false,
    ) where {T <: AbstractArray}
        new{T}(values, creator, grad, generation, name, req_broadcast, CPU())
    end
end

mutable struct CuTensor{T <: CuArray} <: AbstractTensor
    values::T
    creator::Union{Nothing, DiffableFunction}
    grad::Union{Nothing, CuTensor, AbstractArray}
    generation::Int
    name::Union{Nothing, String}
    req_broadcast::Bool
    device::GPU
    function CuTensor(
        values::T;
        creator = nothing,
        grad = nothing,
        generation = 0,
        name = nothing,
        req_broadcast = false,
        device_idx::Int = 0,
    ) where {T <: AbstractArray}
        CUDA.device!(device_idx)
        values = cu(values)
        S = typeof(values)
        new{S}(values, creator, grad, generation, name, req_broadcast, GPU(device_idx))
    end
end


Base.promote(x1::Scalar, x2::T) where {T <: Real} = (x1, Scalar(x2))
Base.promote(x1::T, x2::Scalar) where {T <: Real} = (x1, Scalar(x2))
Base.promote(x1::Scalar, x2::T) where {T <: AbstractArray} = (x1, Tensor(x2))
Base.promote(x1::T, x2::Scalar) where {T <: AbstractArray} = (Tensor(x1), x2)

Base.promote(x1::Tensor, x2::T) where {T <: Real} = (x1, Scalar(x2))
Base.promote(x1::T, x2::Tensor) where {T <: Real} = (Scalar(x1), x2)
Base.promote(x1::Tensor, x2::T) where {T <: AbstractArray} = (x1, Tensor(x2))
Base.promote(x1::T, x2::Tensor) where {T <: AbstractArray} = (Tensor(x1), x2)

Base.promote(x1::CuTensor, x2::T) where {T <: Real} = (x1, Scalar(x2))
Base.promote(x1::T, x2::CuTensor) where {T <: Real} = (Scalar(x1), x2)
Base.promote(x1::CuTensor, x2::T) where {T <: AbstractArray} =
    (x1, CuTensor(x2, device_idx = x1.device.idx))
Base.promote(x1::T, x2::CuTensor) where {T <: AbstractArray} =
    (CuTensor(x1, device_idx = x2.device.idx), x1)


Base.size(x::Variable) = size(x.values)
Base.length(x::Variable) = length(x.values)

function get_output_str(var::Variable)
    output = ""
    output *= "name: $(var.name) \n"
    output *= "values: $(var.values)\n"
    if (var.grad !== nothing)
        output *= "grad: $(var.grad)\n"
    end
    if (var.creator !== nothing)
        output *= "creator: $(repr(var.creator))"
    else
        output *= "creator: User-Defined(nothing)"
    end
    return output
end



function shape_to_out(shape)
    if length(shape) == 1
        return "$(shape[begin])×1"
    end

    out = "$(shape[begin])"

    for element in shape[2:end]
        out *= "×" * string(element) 
    end
    return out
end


# REPL
function Base.show(io::IO, ::MIME"text/plain", x::Scalar)
    type_name = repr(typeof(x.values))
    print(io, "Scalar{$type_name}($(x.values)) \n")
end

# print()
function Base.show(io::IO, x::Scalar)
    print(io, "Scalar($(x.values)) \n")
end

# REPL
function Base.show(io::IO, ::MIME"text/plain", x::Tensor)
    shape = size(x)
    shape_output = shape_to_out(shape)
    type_name = repr(typeof(x.values))
    out = repr("text/plain", x.values)
    value_output = out[findfirst("\n", out)[end]+2:end]
    print(io, "$shape_output Tensor{$type_name}: \n $value_output \n ")
end

# print()
function Base.show(io::IO, x::Tensor)
    print(io, "Tensor($(x.values)) \n")
end

# REPL
function Base.show(io::IO, ::MIME"text/plain", x::CuTensor)
    shape = size(x)
    shape_output = shape_to_out(shape)
    type_name = repr(typeof(x.values))
    out = repr("text/plain", x.values)
    value_output = out[findfirst("\n", out)[end]+2:end]
    print(io, "$shape_output CuTensor{$type_name}: \n $value_output \n ")
end

# print()
function Base.show(io::IO, x::CuTensor)
    print(io, "CuTensor($(x.values)) \n")
end