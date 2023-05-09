struct MapApplyNeg <: UnaryOperator
    grad_field::GradField
end

struct MapApplyMul <: BinaryOperator
    grad_field::GradField
end

struct MapApplyDiv <: BinaryOperator
    grad_field::GradField
end


forward(::Type{MapApplyNeg}, x1::AbstractArray) = -x1

forward(::Type{MapApplyMul}, x1::Real, x2::AbstractArray) = x1 * x2
forward(::Type{MapApplyMul}, x1::AbstractArray, x2::Real) = x1 * x2

forward(::Type{MapApplyDiv}, x1::AbstractArray, x2::Real) = x1 / x2


one_like(::AbstractTensor) = Scalar(1) 
one_like(::AbstractArray) = 1

function _mapapplymul_backward(x1::ScalarTypes, x2::TensorTypes, gy::TensorTypes)
    gx1 = sum(x2) * one_like(gy) 
    gx2 = fill(x1, size(x2)) .* gy
    return gx1, gx2
end


function _mapapplymul_backward(x1::TensorTypes, x2::ScalarTypes, gy::TensorTypes)
    gx1 = fill(x2, size(x1)) .* gy 
    gx2 = sum(x1 .* one_like(gy))
    return gx1, gx2
end


function _mapapplydiv_backward(x1::TensorTypes, x2::ScalarTypes, gy::TensorTypes)
    gx1 = fill(inv(x2.values), size(x1)) .* gy
    gx2 = sum(-x1 / (x2^2) .* gy)
    return gx1, gx2
end



function backward(f::MapApplyNeg, gy::TensorTypes)
    return -gy
end

function backward(f::MapApplyMul, gy::TensorTypes)
    x1, x2 = f.grad_field.inputs
    gx1, gx2 = _mapapplymul_backward(x1, x2, gy)
    return gx1, gx2
end


function backward(f::MapApplyDiv, gy::TensorTypes)
    x1, x2 = f.grad_field.inputs
    gx1, gx2 = _mapapplydiv_backward(x1, x2, gy)
    return gx1, gx2 
end


Base.:-(x::AbstractTensor) = call!(MapApplyNeg, x)

Base.:*(x1::AbstractTensor, x2::Scalar) = call!(MapApplyMul, x1, x2)
Base.:*(x1::Scalar, x2::AbstractTensor) = call!(MapApplyMul, x1, x2)

Base.:*(x1::AbstractTensor, x2::Real) = call!(MapApplyMul, x1, Scalar(x2))
Base.:*(x1::AbstractArray, x2::Scalar) = call!(MapApplyMul, Tensor(x1), x2)

Base.:*(x1::Real, x2::AbstractTensor) = call!(MapApplyMul, Scalar(x1), x2)
Base.:*(x1::Scalar, x2::AbstractArray) = call!(MapApplyMul, x1, Tensor(x2))

Base.:/(x1::AbstractTensor, x2::Scalar) = call!(MapApplyDiv, x1, x2)
Base.:/(x1::AbstractArray, x2::Scalar) = call!(MapApplyDiv, Tensor(x1), x2)
Base.:/(x1::AbstractTensor, x2::Real) = call!(MapApplyDiv, x1, Scalar(x2))


function call!(f::Type{AutoDiff.BroadcastWrapper{MapApplyNeg}}, x::AbstractTensor) 
    x.req_broadcast = false
    call!(MapApplyNeg, x, nograd=nograd)
end


function call!(f::Type{AutoDiff.BroadcastWrapper{MapApplyMul}},  x1::Variable, x2::Variable) 
    x1.req_broadcast = false
    x2.req_broadcast = false
    call!(MapApplyMul, x1, x2, nograd=nograd)
end

function call!(f::Type{AutoDiff.BroadcastWrapper{MapApplyDiv}},  x1::Variable, x2::Variable) 
    x1.req_broadcast = false
    x2.req_broadcast = false
    call!(MapApplyDiv, x1, x2, nograd=nograd)
end