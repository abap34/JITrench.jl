struct GetIndexField{T <: Tuple, S} <: AdditionalField
    in_shape :: T
    index :: S
end

struct GetIndex{T, S} <: DiffableFunction
    grad_field :: GradField
    additional_field :: GetIndexField{T, S}
end

function forward(::Type{GetIndex}, additional_field::GetIndexField, x)
    index = additional_field.index
    return x[index...]
end

function backward(f::GetIndex, gy)
    index = f.additional_field.index
    in_shape = f.additional_field.in_shape
    return nbinary_matrix(in_shape, index, gy)
end

struct NBinaryMatrixField{T, S <: Tuple} <: AdditionalField
    index :: T
    in_shape :: S
end

struct NBinaryMatrix <: UnaryOperator
    grad_field :: GradField
    additional_field :: NBinaryMatrixField
end

function add_at!(A::T, index, val) where T <: AbstractArray
    A[index] .+= val
end

function forward(::Type{NBinaryMatrix}, additional_field::NBinaryMatrixField, x::R) where R <: Real
    index = additional_field.index
    in_shape = additional_field.in_shape
    y = zeros(R, in_shape)
    return add_at!(y, index, x)
end

function backward(f::NBinaryMatrix, gx)
    index = f.additional_field.index
    return gx[index]
end

Base.getindex(x::T, ind...)  where T <: AbstractTensor = call!(GetIndex, GetIndexField(size(x), ind), x)
nbinary_matrix(shape, index, gy) = call!(NBinaryMatrix, NBinaryMatrixField(index, shape), gy)