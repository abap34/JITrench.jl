struct UnaryMapFunction{F} <: UnaryOperator
    grad_field :: GradField
end

forward(::Type{UnaryMapFunction{F}}, x) where F <: DiffableFunction = forward.(F, x)

function backward(f::UnaryMapFunction{F}, gy::T) where {F, T <: TensorTypes}
    x = f.grad_field.inputs[1]
    return backward.(Ref(F), gy) 
end

Base.broadcasted(::typeof(sin), x::Tensor) = call!(UnaryMapFunction{Sin}, x)


``