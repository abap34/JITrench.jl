import ..AutoDiff: broadcast_to
struct SumField{T <: Tuple} <: AdditionalField
    in_shape::T
    dims :: Int
end

struct Sum <: UnaryOperator
    grad_field::GradField
    additional_field::SumField
end

function forward(::Type{Sum}, additional_field::SumField, x)
    if additional_field.dims == -1
        sum(x)
    else
        return sum(x, dims=additional_field.dims)
    end
end

function backward(f::Sum, gy)
    in_shape = f.additional_field.in_shape
    return broadcast_to(gy, in_shape)
end

function Base.sum(x::T; dims=-1) where {T <: AbstractTensor} 
    call!(Sum, SumField(size(x), dims), x)
end