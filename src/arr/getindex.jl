struct GetIndexField{T} <: AdditionalField
    index :: T
end

struct GetIndex{T} <: DiffableFunction
    grad_field :: GradField
    additional_field :: GetIndexField{T}
end

function forward(::Type{GetIndex}, additional_field::GetIndexField, x)
    index = additional_field.index
    return x[index...]
end

function backward(f::GetIndex, gy)
    x = f.grad_field.inputs[1]
    return GetIndexGrad(f.ind, size(x))(gy)
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

function forward(::Type{NBinaryMatrix}, additional_field::NBinaryMatrixField, gy::R) where R <: Real
    index = additional_field.index
    in_shape = additional_field.in_shape
    gx = zeros(R, in_shape)
    return add_at!(gx, index, gy)
end

function backward(f::NBinaryMatrix, gx)
    index = f.additional_field.index
    return gx[index]
end

Base.getindex(x::T, ind...)  where T <: AbstractTensor = call!(GetIndex, GetIndexField(ind), x)
