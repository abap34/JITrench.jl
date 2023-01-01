using CUDA
import Base

abstract type Variable end
abstract type DiffableFunction  end

mutable struct Scalar{T <: Real} <: Variable
    values :: T
    creator :: Union{Nothing, DiffableFunction}
    grad :: Union{Nothing, Scalar, Real}
    generation :: Int
    name :: Union{Nothing, String}
    req_broadcast :: Bool
    device :: CPU
    function Scalar(values::T; creator=nothing, grad=nothing, generation=0, name=nothing, req_broadcast=false) where {T <: Real}
        new{T}(values, creator, grad, generation, name, req_broadcast, CPU())
    end
end

abstract type AbstractTensor <: Variable end

mutable struct Tensor{T <: AbstractArray} <: AbstractTensor
    values :: T
    creator :: Union{Nothing, DiffableFunction}
    grad :: Union{Nothing, Tensor, AbstractArray}
    generation :: Int
    name :: Union{Nothing, String}
    req_broadcast :: Bool
    device :: CPU
    function Tensor(values::T; creator=nothing, grad=nothing, generation=0, name=nothing, req_broadcast=false) where T <: AbstractArray
        new{T}(values, creator, grad, generation, name, req_broadcast, CPU())
    end
end

mutable struct CuTensor{T <: CuArray} <: AbstractTensor
    values :: T
    creator :: Union{Nothing, DiffableFunction}
    grad :: Union{Nothing, CuTensor, AbstractArray}
    generation :: Int
    name :: Union{Nothing, String}
    req_broadcast :: Bool
    device :: GPU
    function CuTensor(values::T; creator=nothing, grad=nothing, generation=0, name=nothing, req_broadcast=false, device_idx::Int=0) where T <: AbstractArray
        CUDA.device!(device_idx)
        values = cu(values)
        S = typeof(values)
        new{S}(values, creator, grad, generation, name, req_broadcast, GPU(device_idx))
    end
end


Base.promote(x1::Scalar, x2::T) where T <: Real = (x1, Scalar(x2))
Base.promote(x1::T, x2::Scalar) where T <: Real = (x1, Scalar(x2))
Base.promote(x1::Scalar, x2::T) where T <: AbstractArray = (x1, Tensor(x2))
Base.promote(x1::T, x2::Scalar) where T <: AbstractArray = (Tensor(x1), x2)

Base.promote(x1::Tensor, x2::T) where T <: Real = (x1, Scalar(x2)) 
Base.promote(x1::T, x2::Tensor) where T <: Real = (Scalar(x1), x2)
Base.promote(x1::Tensor, x2::T) where T <: AbstractArray = (x1, Tensor(x2))
Base.promote(x1::T, x2::Tensor) where T <: AbstractArray = (Tensor(x1), x2)

Base.promote(x1::CuTensor, x2::T) where T <: Real = (x1, Scalar(x2))
Base.promote(x1::T, x2::CuTensor) where T <: Real = (Scalar(x1), x2)
Base.promote(x1::CuTensor, x2::T) where T <: AbstractArray = (x1, CuTensor(x2, device_idx=x1.device.idx))
Base.promote(x1::T, x2::CuTensor) where T <: AbstractArray = (CuTensor(x1, device_idx=x2.device.idx), x1)


Base.size(x::T) where T <: AbstractTensor = size(x.values)

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



# print()
function Base.show(io::IO, var::Variable)
    print(io, "Variable($(var.values))")
end

# REPL
function Base.show(io::IO, ::MIME"text/plain", var::Variable) 
    print(io, get_output_str(var))
end

