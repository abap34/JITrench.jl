struct SumToField{T <: Tuple, S <: Tuple} <: AdditionalField
    in_shape::T
    out_shape::S
end

struct SumTo{T, S} <: UnaryOperator
    grad_field::GradField
    additional_field::SumToField{T, S}
end

function sum_to(x::T, out_shape) where {T <: AbstractArray}
    in_shape = size(x)
    (in_shape == out_shape) && (return x)
    lead = length(in_shape) - length(out_shape)
    if lead == 0
        dims = findall(in_shape .!= out_shape)
        return sum(x, dims = dims)
    elseif lead > 0
        lead_axis = Tuple((length(out_shape) + 1):(length(out_shape) + lead))
        dims = (findall(in_shape[1:(end - lead)] .!= out_shape)..., lead_axis...)
        return dropdims(sum(x, dims = dims), dims = lead_axis)
    else
        # TODO:implement error
    end
end

sum_to(x::AbstractArray, out_shape::Tuple{}) = sum(x)

function forward(::Type{SumTo}, additional_field::SumToField, x)
    return sum_to(x, additional_field.out_shape)
end

function backward(f::SumTo, gy)
    in_shape = f.additional_field.in_shape
    return broadcast_to(gy, in_shape)
end

sum_to(x, shape) = call!(SumTo, SumToField(size(x), shape), x)
