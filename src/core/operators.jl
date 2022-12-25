using .AutoDiff
import .AutoDiff: forward, call!
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


struct Pow{T} <: UnaryOperator
    grad_field::GradField
    c::T
end


@inline forward(::Type{Add}, x1, x2) = x1 + x2
# case: a - b
@inline forward(::Type{Sub}, x1, x2) = x1 - x2
# case: -a
@inline forward(::Type{Sub}, x) = -x
@inline forward(::Type{Neg}, x) = -x
@inline forward(::Type{Mul}, x1, x2) = x1 * x2
@inline forward(::Type{Div}, x1, x2) = x1 / x2
@inline forward(f::Type{Pow}, x1) = x1^f.c

const ScalarTypes = Union{Real, Scalar}
const TensorTypes = Union{AbstractArray, Tensor, CuTensor}

function backward(::Add, gy::T) where T <: Union{ScalarTypes, TensorTypes}
    return (gy, gy)
end

function backward(::Sub, gy::T) where T <: Union{ScalarTypes, TensorTypes}
    return (gy, -gy)
end

function backward(::Neg, gy::T) where T <: Union{ScalarTypes, TensorTypes}
    return -gy
end

function backward(f::Mul, gy::R) where R <: ScalarTypes
    x1, x2 = f.grad_field.inputs
    return (x2 * gy, x1 * gy)
end

function backward(f::Mul, gy::T) where T <: TensorTypes
    x1, x2 = f.grad_field.inputs
    @. return (x2 .* gy, x1 .* gy)
end

function backward(f::Div, gy::R) where R <: ScalarTypes
    x1, x2 = f.grad_field.inputs
    return (1 / x2) * gy, (-x1 / (x2 * x2)) * gy
end

function backward(f::Div, gy::T) where T <: TensorTypes
    x1, x2 = f.grad_field.inputs
    @. return (1 / x2) * gy, (-x1 / (x2 * x2)) * gy
end


function backward(f::Pow, gy::R) where R <: ScalarTypes 
    x = f.grad_field.inputs[1]
    c = f.c
    return (c * (x^(c - 1))) * gy
end


function backward(f::Pow, gy::T) where T <: TensorTypes
    x = f.grad_field.inputs[1]
    c = f.c
    @. return (c * (x^(c - 1))) * gy
end


Base.:+(x1::T, x2::T) where {T <: Scalar} = call!(Add, x1, x2)
Base.:+(x1::T, x2::R) where {T <: Scalar, R <: Real} = call!(Add, x1, Scalar(x2))
Base.:+(x1::R, x2::T) where {T <: Scalar, R <: Real} = call!(Add, Scalar(x1), x2)

Base.:+(x1::T, x2::T) where {T <: Tensor} = call!(Add, x1, x2)
Base.:+(x1::T, x2::S) where {T <: Tensor, S <: AbstractArray} = call!(Add, x1, Tensor(x2))
Base.:+(x1::S, x2::T) where {T <: Tensor, S <: AbstractArray} = call!(Add, Tensor(x1), x2)

Base.:+(x1::T, x2::T) where {T <: CuTensor} = call!(Add, x1, x2)
Base.:+(x1::T, x2::S) where {T <: CuTensor, S <: AbstractArray} = NotSameDeviceError(same_accelerator=false, same_gpu_idx=false)
Base.:+(x1::S, x2::T) where {T <: CuTensor, S <: AbstractArray} = NotSameDeviceError(same_accelerator=false, same_gpu_idx=false)


Base.:-(x1::T, x2::T) where {T <: Scalar} = call!(Sub, x1, x2)
Base.:-(x1::T, x2::R) where {T <: Scalar, R <: Real} = call!(Sub, x1, Scalar(x2))
Base.:-(x1::R, x2::T) where {T <: Scalar, R <: Real} = call!(Sub, Scalar(x1), x2)

Base.:-(x1::T, x2::T) where {T <: Tensor} = call!(Sub, x1, x2)
Base.:-(x1::T, x2::S) where {T <: Tensor, S <: AbstractArray} = call!(Sub, x1, Tensor(x2))
Base.:-(x1::S, x2::T) where {T <: Tensor, S <: AbstractArray} = call!(Sub, Tensor(x1), x2)

Base.:-(x1::T, x2::T) where {T <: CuTensor} = call!(Sub, x1, x2)
Base.:-(x1::T, x2::S) where {T <: CuTensor, S <: AbstractArray} = NotSameDeviceError(same_accelerator=false, same_gpu_idx=false)
Base.:-(x1::S, x2::T) where {T <: CuTensor, S <: AbstractArray} = NotSameDeviceError(same_accelerator=false, same_gpu_idx=false)

Base.:-(x::T) where {T <: Variable} = call!(Neg, x)



Base.:*(x1::T, x2::T) where {T <: Scalar} = call!(Mul, x1, x2)
Base.:*(x1::T, x2::R) where {T <: Scalar, R <: Real} = call!(Mul, x1, Scalar(x2))
Base.:*(x1::R, x2::T) where {T <: Scalar, R <: Real} = call!(Mul, Scalar(x1), x2)

Base.:*(x1::T, x2::T) where {T <: Tensor} = call!(Mul, x1, x2)
Base.:*(x1::T, x2::S) where {T <: Tensor, S <: AbstractArray} = call!(Mul, x1, Tensor(x2))
Base.:*(x1::S, x2::T) where {T <: Tensor, S <: AbstractArray} = call!(Mul, Tensor(x1), x2)

Base.:*(x1::T, x2::T) where {T <: CuTensor} = call!(Mul, x1, x2)
Base.:*(x1::T, x2::S) where {T <: CuTensor, S <: AbstractArray} = NotSameDeviceError(same_accelerator=false, same_gpu_idx=false)
Base.:*(x1::S, x2::T) where {T <: CuTensor, S <: AbstractArray} = NotSameDeviceError(same_accelerator=false, same_gpu_idx=false)

Base.:*(x1::T, x2::R) where {T <: Tensor, R <: Real} = call!(Mul, x1, Scalar(x2))
Base.:*(x1::R, x2::T) where {T <: Tensor, R <: Real} = call!(Mul, Scalar(x1), x2)

Base.:*(x1::T, x2::R) where {T <: CuTensor, R <: Real} = call!(Mul, x1, Scalar(x2))
Base.:*(x1::R, x2::T) where {T <: CuTensor, R <: Real} = call!(Mul, Scalar(x1), x2)



Base.:/(x1::T, x2::T) where {T <: Scalar} = call!(Div, x1, x2)
Base.:/(x1::T, x2::R) where {T <: Scalar, R <: Real} = call!(Div, x1, Scalar(x2))
Base.:/(x1::R, x2::T) where {T <: Scalar, R <: Real} = call!(Div, Scalar(x1), x2)

Base.:/(x1::T, x2::T) where {T <: Tensor} = call!(Div, x1, x2)
Base.:/(x1::T, x2::S) where {T <: Tensor, S <: AbstractArray} = call!(Div, x1, Tensor(x2))
Base.:/(x1::S, x2::T) where {T <: Tensor, S <: AbstractArray} = call!(Div, Tensor(x1), x2)

Base.:/(x1::T, x2::T) where {T <: CuTensor} = call!(Div, x1, x2)
Base.:/(::T, ::S) where {T <: CuTensor, S <: AbstractArray} = NotSameDeviceError(same_accelerator=false, same_gpu_idx=false)
Base.:/(::S, ::T) where {T <: CuTensor, S <: AbstractArray} = NotSameDeviceError(same_accelerator=false, same_gpu_idx=false)

Base.:/(x1::T, x2::R) where {T <: Tensor, R <: Real} = call!(Div, x1, Scalar(x2))
Base.:/(x1::R, x2::T) where {T <: Tensor, R <: Real} = call!(Div, Scalar(x1), x2)

Base.:/(x1::T, x2::R) where {T <: CuTensor, R <: Real} = call!(Div, x1, Scalar(x2))
Base.:/(x1::R, x2::T) where {T <: CuTensor, R <: Real} = call!(Div, Scalar(x1), x2)


function AutoDiff.call!(F::Type{Pow}, x::T, c::R) where {T<:Variable, R <: Real}
    inputs = (x, )
    y = x.values ^ c
    gen = x.generation
    gf = GradField(
            inputs,
            AutoDiff.out_to_tensor(y, gen),
            gen
        ) 
    func = Pow(gf, c)
    gf.output.creator = func
    return gf.output
end


Base.:^(x1::T, x2::T) where {T <: Scalar} = call!(Pow, x1, x2.values)
Base.:^(x1::T, x2::R) where {T <: Scalar, R <: Real} = call!(Pow, x1, x2)
Base.:^(x1::R, x2::T) where {T <: Scalar, R <: Real} = call!(Pow, Scalar(x1), x2.values)

