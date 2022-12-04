using CUDA

abstract type Variable end
abstract type DiffableFunction  end

mutable struct Scalar{T <: Real} <: Variable
    values :: T
    creator :: Union{Nothing, DiffableFunction}
    grad :: Union{Nothing, Scalar}
    generation :: Int
    name :: Union{Nothing, String}
    req_broadcast :: Bool
    function Scalar(values::T; creator=nothing, grad=nothing, generation=0, name=nothing, req_broadcast=false) where {T <: Real}
        new{T}(values, creator, grad, generation, name, req_broadcast)
    end
end

abstract type AbstractTensor <: Variable end

mutable struct Tensor{T <: AbstractArray} <: AbstractTensor
    values :: T
    creator :: Union{Nothing, DiffableFunction}
    grad :: Union{Nothing, Scalar}
    generation :: Int
    name :: Union{Nothing, String}
    req_broadcast :: Bool
    function Tensor(values::T; creator=nothing, grad=nothing, generation=0, name=nothing, req_broadcast=false) where T <: AbstractArray
        new{T}(values, creator, grad, generation, name, req_broadcast)
    end
end

mutable struct CuTensor{T <: CuArray} <: AbstractTensor
    values :: T
    creator :: Union{Nothing, DiffableFunction}
    grad :: Union{Nothing, Scalar}
    generation :: Int
    name :: Union{Nothing, String}
    req_broadcast :: Bool
    device :: GPU
    function CuTensor(values::T; creator=nothing, grad=nothing, generation=0, name=nothing, req_broadcast=false, device_idx::Int=0) where T <: AbstractArray
        CUDA.device!(device_idx)
        values = cu(values)
        S = typeof(values)
        device = 
        new{S}(values, creator, grad, generation, name, req_broadcast, GPU(device_idx))
    end
end


Base.promote_rule(::Type{<:Real}, ::Type{<:Variable}) = Variable

Base.convert(::Type{Variable}, x::AbstractArray) = Variable(x)

Base.convert(::Type{Variable}, x::Real) = Variable(x)

Base.convert(::Type{Variable}, x::Variable) = x

Base.size(x::Variable) = size(x.values)

function get_output_str(var::Variable)
    output = ""
    output *= "name: $(var.name) \n"
    output *= "values: $(var.values)\n"
    if (var.grad !== nothing) 
        output *= "grad: $(var.grad)\n"
    end
    if (var.creator !== nothing)
        output *= "creator: $(typeof(var.creator))"
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

