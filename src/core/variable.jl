import Base

abstract type TrenchObject end

mutable struct Variable <: TrenchObject
    values 
    creator
    grad 
    generation 
    name 
    function Variable(values; creator=nothing, grad=nothing, generation=0, name=nothing)      
        new(values, creator, grad, generation, name)
    end
end

Base.promote_rule(::Type{<:Real}, ::Type{Variable}) = Variable

Base.convert(::Type{Variable}, x::AbstractArray) = Variable(x)

Base.convert(::Type{Variable}, x::Real) = Variable(x)

Base.convert(::Type{Variable}, x::Variable) = x
