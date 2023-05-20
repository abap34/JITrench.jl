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



Base.:-(x1::Scalar, x2::Scalar) = call!(Sub, x1, x2)
Base.:-(x1::Scalar, x2::Real) = call!(Sub, x1, Scalar(x2))
Base.:-(x1::Real, x2::Scalar) = call!(Sub, Scalar(x1), x2)

Base.:-(x1::Tensor, x2::Tensor) = call!(Sub, x1, x2)
Base.:-(x1::Tensor, x2::AbstractArray) = call!(Sub, x1, Tensor(x2))
Base.:-(x1::AbstractArray, x2::Tensor) = call!(Sub, Tensor(x1), x2)

Base.:-(x1::CuTensor, x2::CuTensor) = call!(Sub, x1, x2)
Base.:-(::CuTensor, ::AbstractArray) =
    NotSameDeviceError(same_accelerator = false, same_gpu_idx = false)
Base.:-(::S, ::T) where {T <: CuTensor, S <: AbstractArray} =
    NotSameDeviceError(same_accelerator = false, same_gpu_idx = false)

Base.:-(x::Variable) = call!(Neg, x)


Base.:-(x1::AbstractTensor, x2::Real) = call!(Sub, x1, Scalar(x2))
Base.:-(x1::Real, x2::AbstractTensor) = call!(Sub, Scalar(x1), x2)
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


Base.:*(x1::Scalar, x2::Scalar) = call!(Mul, x1, x2)
Base.:*(x1::Scalar, x2::Real) = call!(Mul, x1, Scalar(x2))
Base.:*(x1::Real, x2::Scalar) = call!(Mul, Scalar(x1), x2)

Base.:*(x1::Tensor, x2::Tensor) = call!(Mul, x1, x2)
Base.:*(x1::Tensor, x2::AbstractArray) = call!(Mul, x1, Tensor(x2))
Base.:*(x1::AbstractArray, x2::Tensor) = call!(Mul, Tensor(x1), x2)

Base.:*(x1::CuTensor, x2::CuTensor) = call!(Mul, x1, x2)
Base.:*(::CuTensor, ::AbstractArray) =
    NotSameDeviceError(same_accelerator = false, same_gpu_idx = false)
Base.:*(::AbstractArray, ::CuTensor) =
    NotSameDeviceError(same_accelerator = false, same_gpu_idx = false)

Base.:/(x1::Scalar, x2::Scalar) = call!(Div, x1, x2)
Base.:/(x1::Scalar, x2::Real) = call!(Div, x1, Scalar(x2))
Base.:/(x1::Real, x2::Scalar) = call!(Div, Scalar(x1), x2)

Base.:/(x1::Tensor, x2::Tensor) = call!(Div, x1, x2)
Base.:/(x1::Tensor, x2::AbstractArray) = call!(Div, x1, Tensor(x2))
Base.:/(x1::AbstractArray, x2::Tensor) = call!(Div, Tensor(x1), x2)

Base.:/(x1::CuTensor, x2::CuTensor) = call!(Div, x1, x2)
Base.:/(::CuTensor, ::AbstractArray) =
    NotSameDeviceError(same_accelerator = false, same_gpu_idx = false)
Base.:/(::AbstractArray, ::CuTensor) =
    NotSameDeviceError(same_accelerator = false, same_gpu_idx = false)

function Base.:/(x1::AbstractTensor, x2::AbstractTensor)
    if check_broadcastable(x1, x2)
        call!(Div, x1, x2)
    else
        throw(BroadcastCallError())
    end
end

Base.:^(x1::Scalar, x2::Scalar) = call!(Pow, PowField(x2.values), x1)
Base.:^(x1::Scalar, x2::Real) = call!(Pow, PowField(x2), x1)
Base.:^(x1::Real, x2::Scalar) =
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
        if x2.values == 2
            call!(Square, x1)
        else
            call!(Pow, PowField(x2), x1, x2)
        end
    else
        throw(BroadcastCallError())
    end
end


function Base.:^(x1::Scalar, x2::AbstractTensor)
    x1 = Scalar(x1)
    if check_broadcastable(x1, x2)
        if x1.values == 2
            call!(Square, x2)
        else
            call!(Pow, PowField(x2), x1, x2)
        end
    else
        throw(BroadcastCallError())
    end
end



struct Sin <: UnaryOperator
    grad_field::GradField
end

struct Cos <: UnaryOperator
    grad_field::GradField
end

struct Tan <: UnaryOperator
    grad_field::GradField
end

struct Log <: UnaryOperator
    grad_field::GradField
end

struct Exp <: UnaryOperator
    grad_field::GradField
end

struct Square <: UnaryOperator
    grad_field::GradField
end

struct Sqrt <: UnaryOperator
    grad_field::GradField
end

struct Inv <: UnaryOperator
    grad_field::GradField
end

@inline forward(::Type{Sin}, x) = sin(x)
@inline forward(::Type{Cos}, x) = cos(x)
@inline forward(::Type{Tan}, x) = tan(x)
@inline forward(::Type{Log}, x) = log(x)
@inline forward(::Type{Exp}, x) = exp(x)
@inline forward(::Type{Square}, x) = x^2
@inline forward(::Type{Sqrt}, x) = sqrt(x)
@inline forward(::Type{Inv}, x) = inv(x)

function backward(f::Sin, gy::ScalarTypes)
    x = f.grad_field.inputs[1]
    return cos(x) * gy
end

function backward(f::Sin, gy::TensorTypes)
    x = f.grad_field.inputs[1]
    @. return cos(x) * gy
end

function backward(f::Cos, gy::ScalarTypes)
    x = f.grad_field.inputs[1]
    return -sin(x) * gy
end

function backward(f::Cos, gy::TensorTypes)
    x = f.grad_field.inputs[1]
    @. return -sin(x) * gy
end

function backward(f::Tan, gy::ScalarTypes)
    x = f.grad_field.inputs[1]
    return inv(cos(x)^2) * gy
end

function backward(f::Tan, gy::TensorTypes)
    x = f.grad_field.inputs[1]
    @. return inv(cos(x)^2) * gy
end


function backward(f::Log, gy::ScalarTypes)
    x = f.grad_field.inputs[1]
    return gy / x
end

function backward(f::Log, gy::TensorTypes)
    x = f.grad_field.inputs[1]
    @. return gy / x
end


function backward(f::Exp, gy::ScalarTypes)
    x = f.grad_field.inputs[1]
    return exp(x) * gy
end

function backward(f::Exp, gy::TensorTypes)
    x = f.grad_field.inputs[1]
    @. return exp(x) * gy
end

function backward(f::Square, gy::ScalarTypes)
    x = f.grad_field.inputs[1]
    return 2x * gy
end

function backward(f::Square, gy::TensorTypes)
    x = f.grad_field.inputs[1]
    @. return 2x * gy
end


function backward(f::Sqrt, gy::ScalarTypes)
    x = f.grad_field.inputs[1]
    return inv(2*sqrt(x)) * gy
end

function backward(f::Sqrt, gy::TensorTypes)
    x = f.grad_field.inputs[1]
    @. return inv(2*sqrt(x)) * gy
end

function backward(f::Inv, gy::ScalarTypes)
    x = f.grad_field.inputs[1]
    return (-gy /(x^2))
end

function backward(f::Inv, gy::TensorTypes)
    x = f.grad_field.inputs[1]
    return (-gy ./ (x.^2)) 
end

Base.sin(x::Scalar) = call!(Sin, x)
Base.cos(x::Scalar) = call!(Cos, x)
Base.tan(x::Scalar) = call!(Tan, x)
Base.log(x::Scalar) = call!(Log, x)
Base.exp(x::Scalar) = call!(Exp, x)
Base.sqrt(x::Scalar) = call!(Sqrt, x)
Base.inv(x::Scalar) = call!(Inv, x)

function Base.sin(x::AbstractTensor)
    check_broadcastable(x)
    call!(Sin, x)
end

function Base.cos(x::AbstractTensor)
    check_broadcastable(x)
    call!(Cos, x)
end

function Base.tan(x::AbstractTensor)
    check_broadcastable(x)
    call!(Tan, x)
end

function Base.log(x::AbstractTensor)
    check_broadcastable(x)
    call!(Log, x)
end

function Base.exp(x::AbstractTensor) 
    check_broadcastable(x)
    call!(Exp, x)
end

function Base.sqrt(x::AbstractTensor)
    check_broadcastable(x)
    call!(Sqrt, x)
end

function Base.inv(x::AbstractTensor)
    check_broadcastable(x)
    call!(Inv, x)
end


Base.literal_pow(::typeof(^), x::Scalar, ::Val{2}) = call!(Square, x)

function Base.literal_pow(::typeof(^), x::AbstractTensor, ::Val{2})  
    check_broadcastable(x)
    call!(Square, x)
end


function Base.broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::AbstractTensor, ::Val{c}) where c 
    if c == 2
        x.req_broadcast = true
        call!(Square, x)
    else
        x.req_broadcast = true
        call!(Pow, PowField(c), x)
    end
end


function Base.broadcasted(::typeof(^), x::Variable, c)
    if c == 2
        x.req_broadcast = true
        call!(Square, x)
    else
        x.req_broadcast = true
        call!(Pow, PowField(c), x)
    end
end






using .AutoDiff
import .AutoDiff: forward, backward, call!
import Base

struct ClampField{S <: Number, T <: Number} <: AdditionalField
    lo :: S
    hi :: T
end

struct Clamp{S, T} <: UnaryOperator
    grad_field :: GradField
    additional_field :: ClampField{S, T}
end

function forward(::Type{Clamp}, clamp_field::ClampField, x)
    return Base.clamp(x, clamp_field.lo, clamp_field.hi)
end

function backward(f::Clamp, gy::ScalarTypes)
    x = f.grad_field.inputs[1]
    return Int((x.values >= f.additional_field.lo) && (x.values <= f.additional_field.hi)) * gy
end

function backward(f::Clamp, gy::TensorTypes)
    x = f.grad_field.inputs[1]
    @. return Int((x.values >= f.additional_field.lo) && (x.values <= f.additional_field.hi)) * gy
end

Base.clamp(x::Scalar, lo, hi) = call!(Clamp, ClampField(lo, hi), x)

function Base.clamp(x::AbstractTensor, lo, hi) 
    call!(Clamp, ClampField(lo, hi), x)
end
