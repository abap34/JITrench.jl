
abstract type AdditionalField end

"""
    GradField(inputs, output, generation)

Retain the information of the function for backpropagation.
All DiffableFunction must have this field.
"""
struct GradField{T, S}
    " Tuple of input variables."
    inputs::T
    " Output variable."
    output::S
    " Generation of the function. Corresponds evaluation order priority of the function in backward pass. "
    generation::Int
    function GradField(
        inputs::T,
        output::S,
        generation::Int,
    ) where {T <: Tuple, S <: Variable}
        new{T, S}(inputs, output, generation)
    end
end

"""
    BinaryOperator

DiffableFunction which takes two variables as input.
"""
abstract type BinaryOperator <: DiffableFunction end

"""
    UnaryOperator

DiffableFunction which takes one variable as input.
"""
abstract type UnaryOperator <: DiffableFunction end


function Base.show(io::IO, f::DiffableFunction)
    print(io, typeof(f))
end


"""
    _get_gf(f::DiffableFunction)

Get the GradField of the function.
"""
_get_gf(f::DiffableFunction) = f.grad_field


"""
    @diffable

"""



function Base.show(io::IO, ::MIME"text/plain", f::DiffableFunction)
    print(io, typeof(f))
end


function _subtypedef(ex)
    if ex.head != :<:
        return false
    end

    if ex.args[2] isa Symbol
        return eval(ex.args[2]) <: DiffableFunction
    else
        return false
    end
end

"""
    @diffable

Check definition of DiffableFunction has `grad_field` field and it's type is GradField.
"""
macro diffable(ex)
    if (ex.head != :struct)
        throw(ArgumentError(
            "@diffable is macro that check definition of struct. Passed argument is not struct."
        ))
    end
    if (ex.args[2].head != :<:) || !(JITrench.AutoDiff.eval(ex.args[2].args[2]) <: DiffableFunction)
        throw(ArgumentError(
            "@diffable is macro that check definition of struct. Which is subtypes of DiffableFunction."
        ))
    end

    if !(ex.args[3] isa Expr)
        throw(ArgumentError(
            "@diffable is macro that check definition of struct. Which has grad_field field."
        ))
    end
    
    for arg in ex.args[3].args
        if arg isa Expr 
            if arg.head == :(::)
                if arg.args[1] == :grad_field
                    if JITrench.AutoDiff.eval(arg.args[2]) <: GradField
                        return esc(ex)
                    else
                        throw(ArgumentError(
                            "Type of grad_field field must be GradField."
                        ))
                    end
                end
            end
        end
    end
    throw(ArgumentError(
        "DiffableFunction must have `grad_field` field."
    ))
end

