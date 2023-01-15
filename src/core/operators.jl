using .AutoDiff
import .AutoDiff: forward, backward, call!
import Base

struct Add <: BinaryOperator
    grad_field::GradField
end

struct Sub <: BinaryOperator
    grad_field::GradField
end

struct Neg <: UnaryOperator
    grad_field::GradField
end

struct Mul <: BinaryOperator
    grad_field::GradField
end


struct Div <: BinaryOperator
    grad_field::GradField
end

struct PowField{T} <: AdditionalField
    c::T
end

struct Pow{T} <: UnaryOperator
    grad_field::GradField
    additional_field::PowField{T}
end



@inline forward(::Type{Add}, x1, x2) = x1 + x2
@inline forward(::Type{Sub}, x1, x2) = x1 - x2
@inline forward(::Type{Sub}, x) = -x
@inline forward(::Type{Neg}, x) = -x
@inline forward(::Type{Mul}, x1, x2) = x1 * x2
@inline forward(::Type{Div}, x1, x2) = x1 / x2
@inline forward(::Type{Pow}, pow_field::PowField, x1) = x1^(pow_field.c)

function backward(::Add, gy::Union{ScalarTypes, TensorTypes})  
    return (gy, gy)
end

function backward(::Sub, gy::Union{ScalarTypes, TensorTypes})  
    return (gy, -gy)
end

function backward(::Neg, gy::Union{ScalarTypes, TensorTypes})  
    return -gy
end

function backward(f::Mul, gy::ScalarTypes)
    x1, x2 = f.grad_field.inputs
    return (x2 * gy, x1 * gy)
end

function backward(f::Mul, gy::TensorTypes) 
    x1, x2 = f.grad_field.inputs
    @. return (x2 * gy, x1 * gy)
end

function backward(f::Div, gy::ScalarTypes)
    x1, x2 = f.grad_field.inputs
    return inv(x2) * gy, (-x1 / (x2^2)) * gy
end

function backward(f::Div, gy::TensorTypes) 
    x1, x2 = f.grad_field.inputs
    @. return inv(x2) * gy, (-x1 / (x2^2)) * gy
end


function backward(f::Pow, gy::ScalarTypes)
    x = f.grad_field.inputs[1]
    c = f.additional_field.c
    return (c * (x^(c - 1))) * gy
end


function backward(f::Pow, gy::TensorTypes) 
    x = f.grad_field.inputs[1]
    c = f.additional_field.c
    @. return (c * (x^(c - 1))) * gy
end


Base.:+(x1::Scalar, x2::Scalar) = call!(Add, x1, x2)
Base.:+(x1::Scalar, x2::Real)   = call!(Add, x1, Scalar(x2))
Base.:+(x1::Real, x2::Scalar)   = call!(Add, Scalar(x1), x2)

Base.:+(x1::T, x2::S) where {T <: Tensor, S <: AbstractArray} = call!(Add, x1, Tensor(x2))
Base.:+(x1::S, x2::T) where {T <: Tensor, S <: AbstractArray} = call!(Add, Tensor(x1), x2)

Base.:+(x1::T, x2::T) where {T <: CuTensor} = call!(Add, x1, x2)
Base.:+(::T, ::S) where {T <: CuTensor, S <: AbstractArray} =
    NotSameDeviceError(same_accelerator = false, same_gpu_idx = false)
Base.:+(::S, ::T) where {T <: CuTensor, S <: AbstractArray} =
    NotSameDeviceError(same_accelerator = false, same_gpu_idx = false)

Base.:+(x1::AbstractTensor, x2::Real) = call!(Add, x1, Scalar(x2))
Base.:+(x1::Real, x2::AbstractArray) = call!(Add, Scalar(x1), x2)
Base.:+(x1::AbstractArray, x2::Scalar) = call!(Add, Tensor(x1), x2)
Base.:+(x1::Scalar, x2::AbstractArray) = call!(Add, x1, Tensor(x2))

function Base.:+(x1::AbstractTensor, x2::AbstractTensor)
    if check_sameshape(x1, x2)
        call!(Add, x1, x2)
    else
        if check_broadcastable(x1, x2)
            call!(Add, x1, x2)
        else
            throw(BroadcastCallError())
        end
    end
end

function Base.:+(x1::AbstractTensor, x2::Scalar)
    if check_broadcastable(x1, x2)
        call!(Add, x1, x2)
    else
        throw(BroadcastCallError())
    end
end


function Base.:+(x1::Scalar, x2::AbstractTensor)
    if check_broadcastable(x1, x2)
        call!(Add, x1, x2)
    else
        throw(BroadcastCallError())
    end
end



Base.:-(x1::T, x2::T) where {T <: Scalar} = call!(Sub, x1, x2)
Base.:-(x1::T, x2::R) where {T <: Scalar, R <: Real} = call!(Sub, x1, Scalar(x2))
Base.:-(x1::R, x2::T) where {T <: Scalar, R <: Real} = call!(Sub, Scalar(x1), x2)

Base.:-(x1::T, x2::T) where {T <: Tensor} = call!(Sub, x1, x2)
Base.:-(x1::T, x2::S) where {T <: Tensor, S <: AbstractArray} = call!(Sub, x1, Tensor(x2))
Base.:-(x1::S, x2::T) where {T <: Tensor, S <: AbstractArray} = call!(Sub, Tensor(x1), x2)

Base.:-(x1::T, x2::T) where {T <: CuTensor} = call!(Sub, x1, x2)
Base.:-(::T, ::S) where {T <: CuTensor, S <: AbstractArray} =
    NotSameDeviceError(same_accelerator = false, same_gpu_idx = false)
Base.:-(::S, ::T) where {T <: CuTensor, S <: AbstractArray} =
    NotSameDeviceError(same_accelerator = false, same_gpu_idx = false)

Base.:-(x::T) where {T <: Variable} = call!(Neg, x)


Base.:-(x1::AbstractTensor, x2::Real) = call!(Sub, x1, Scalar(x2))
Base.:-(x1::Real, x2::AbstractArray) = call!(Sub, Scalar(x1), x2)
Base.:-(x1::AbstractArray, x2::Scalar) = call!(Sub, Tensor(x1), x2)
Base.:-(x1::Scalar, x2::AbstractArray) = call!(Sub, x1, Tensor(x2))

function Base.:-(x1::AbstractTensor, x2::AbstractTensor)
    if check_sameshape(x1, x2)
        call!(Sub, x1, x2)
    else
        if check_broadcastable(x1, x2)
            call!(Sub, x1, x2)
        else
            throw(BroadcastCallError())
        end
    end
end

function Base.:-(x1::AbstractTensor, x2::Scalar)
    if check_broadcastable(x1, x2)
        call!(Sub, x1, x2)
    else
        throw(BroadcastCallError())
    end
end


function Base.:-(x1::Scalar, x2::AbstractTensor)
    if check_broadcastable(x1, x2)
        call!(Sub, x1, x2)
    else
        throw(BroadcastCallError())
    end
end


Base.:*(x1::T, x2::T) where {T <: Scalar} = call!(Mul, x1, x2)
Base.:*(x1::T, x2::R) where {T <: Scalar, R <: Real} = call!(Mul, x1, Scalar(x2))
Base.:*(x1::R, x2::T) where {T <: Scalar, R <: Real} = call!(Mul, Scalar(x1), x2)

Base.:*(x1::T, x2::T) where {T <: Tensor} = call!(Mul, x1, x2)
Base.:*(x1::T, x2::S) where {T <: Tensor, S <: AbstractArray} = call!(Mul, x1, Tensor(x2))
Base.:*(x1::S, x2::T) where {T <: Tensor, S <: AbstractArray} = call!(Mul, Tensor(x1), x2)

Base.:*(x1::T, x2::T) where {T <: CuTensor} = call!(Mul, x1, x2)
Base.:*(::T, ::S) where {T <: CuTensor, S <: AbstractArray} =
    NotSameDeviceError(same_accelerator = false, same_gpu_idx = false)
Base.:*(::S, ::T) where {T <: CuTensor, S <: AbstractArray} =
    NotSameDeviceError(same_accelerator = false, same_gpu_idx = false)

Base.:/(x1::T, x2::T) where {T <: Scalar} = call!(Div, x1, x2)
Base.:/(x1::T, x2::R) where {T <: Scalar, R <: Real} = call!(Div, x1, Scalar(x2))
Base.:/(x1::R, x2::T) where {T <: Scalar, R <: Real} = call!(Div, Scalar(x1), x2)

Base.:/(x1::T, x2::T) where {T <: Tensor} = call!(Div, x1, x2)
Base.:/(x1::T, x2::S) where {T <: Tensor, S <: AbstractArray} = call!(Div, x1, Tensor(x2))
Base.:/(x1::S, x2::T) where {T <: Tensor, S <: AbstractArray} = call!(Div, Tensor(x1), x2)

Base.:/(x1::T, x2::T) where {T <: CuTensor} = call!(Div, x1, x2)
Base.:/(::T, ::S) where {T <: CuTensor, S <: AbstractArray} =
    NotSameDeviceError(same_accelerator = false, same_gpu_idx = false)
Base.:/(::S, ::T) where {T <: CuTensor, S <: AbstractArray} =
    NotSameDeviceError(same_accelerator = false, same_gpu_idx = false)

function Base.:/(x1::AbstractTensor, x2::AbstractTensor)
    if check_broadcastable(x1, x2)
        call!(Div, x1, x2)
    else
        throw(BroadcastCallError())
    end
end

Base.:^(x1::T, x2::T) where {T <: Scalar} = call!(Pow, PowField(x2.values), x1)
Base.:^(x1::T, x2::R) where {T <: Scalar, R <: Real} = call!(Pow, PowField(x2), x1)
Base.:^(x1::R, x2::T) where {T <: Scalar, R <: Real} =
    call!(Pow, PowField(x2.values), Scalar(x1))


function Base.:^(x1::AbstractTensor, x2::AbstractTensor)
    if check_broadcastable(x1, x2)
        call!(Pow, PowField(x2.values), x1, x2)
    else
        throw(BroadcastCallError())
    end
end

function Base.:^(x1::AbstractTensor, x2::Scalar)
    if check_broadcastable(x1, x2)
        call!(Pow, PowField(x2), x1, x2)
    else
        throw(BroadcastCallError())
    end
end


function Base.:^(x1::Scalar, x2::AbstractTensor)
    x1 = Scalar(x1)
    if check_broadcastable(x1, x2)
        call!(Pow, PowField(x2), x1, x2)
    else
        throw(BroadcastCallError())
    end
end