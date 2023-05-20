struct MeanSquaredError <: BinaryOperator
    grad_field :: GradField
end

forward(::Type{MeanSquaredError}, x1, x2) = sum((x1 - x2).^2) / length(x1)

function backward(f::MeanSquaredError, gy::ScalarTypes)
    x1, x2 = f.grad_field.inputs
    diff = x1 - x2
    gx1 = (fill(2gy, size(diff)) .* diff) / length(diff)
    return gx1, -gx1
end

mean_squared_error(x1::AbstractTensor, x2::AbstractTensor) = call!(MeanSquaredError, x1, x2)


function logsumexp(x::AbstractArray, dims=1)
    m = maximum(x, dims=dims)
    y = x .- m
    y .= exp.(y)
    s = sum(y, dims=dims)
    s = log.(s)
    m .+= s
    return m
end


function one_hot_batch(labels::AbstractArray, nclass::Int)
    N = size(labels)[1]
    result = zeros(Bool, N, nclass)
    for (i, label) in enumerate(labels)
        result[i, label] = 1
    end
    return result
end

function one_hot_batch(labels::Tensor, nclass::Int)
    return Tensor(one_hot_batch(labels.values, nclass))
end


struct SoftmaxCrossEntropyField{T} <: AdditionalField
    t :: T
end

struct SoftmaxCrossEntropy <: UnaryOperator
    grad_field :: GradField
    additional_field :: SoftmaxCrossEntropyField
end

function forward(::Type{SoftmaxCrossEntropy}, additional_field::SoftmaxCrossEntropyField, x)
    t = additional_field.t
    N = size(x)[1]
    return sum((logsumexp(x, 2) - getindex.(eachrow(x), t))) / N
end


function backward(f::SoftmaxCrossEntropy, gy::ScalarTypes)
    x = f.grad_field.inputs[1]
    t = f.additional_field.t
    N, p = size(x)
    y = softmax(x)
    t_onehot = one_hot_batch(t, p)
    return (y - t_onehot) .* (gy / N)
end

softmax_cross_entropy(x::AbstractTensor, t::AbstractTensor) = call!(SoftmaxCrossEntropy, SoftmaxCrossEntropyField(t.values), x)