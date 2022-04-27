"""
    Variable(values, [creator, grad, generation, name])

A type that is a variable in JITrench. 

For a detailed explanation of this, see the documentation(here).

Only real numbers or arrays consisting of real numbers can be stored.

# Examples
```julia-repl
julia> Variable(2)
name: nothing 
values: 2
creator: User-Defined(nothing)

julia> Variable([1.2, 3.5])
name: nothing 
values: [1.2, 3.5]
creator: User-Defined(nothing)
```

"""
mutable struct Variable{T} <: TrenchObject
    values::T
    creator :: Union{Nothing, DiffableFunction}
    grad::Union{Nothing, Variable}
    generation::Int
    name::Union{Nothing,String}
    req_broadcast :: Bool
    function Variable(values::T; creator=nothing, grad=nothing, generation=0, name=nothing, req_broadcast=false) where {T <: Union{<:Real,AbstractArray{<:Real}}}
        new{T}(values, creator, grad, generation, name, req_broadcast)
    end
end

Base.promote_rule(::Type{<:Real}, ::Type{<:Variable}) = Variable

Base.convert(::Type{Variable}, x::AbstractArray) = Variable(x)

Base.convert(::Type{Variable}, x::Real) = Variable(x)

Base.convert(::Type{Variable}, x::Variable) = x

Base.size(x::Variable) = size(x.values)

function get_output_str(var::Variable)
    output = ""
    output *= "name: $(var.name) \n"
    output *= "values: $(var.values)\n"
    if (var.grad !== nothing) 
        output *= "grad: $(var.grad)\n"
    end
    if (var.creator !== nothing)
        output *= "creator: $(typeof(var.creator))"
    else
        output *= "creator: User-Defined(nothing)"
    end
    return output
end


# print()
function Base.show(io::IO, var::Variable)
    print(io, "Variable($(var.values))")
end

# REPL
function Base.show(io::IO, ::MIME"text/plain", var::Variable) 
    print(io, get_output_str(var))
end

