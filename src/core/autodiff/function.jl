abstract type AdditionalField end

struct GradField{T, S} 
    inputs :: T
    output :: S
    generation::Int
    function GradField(inputs::T, output::S, generation::Int) where {T <: Tuple, S <: Variable}
        new{T, S}(inputs, output, generation)
    end
end

abstract type BinaryOperator <: DiffableFunction end
abstract type UnaryOperator <: DiffableFunction end

function Base.show(io::IO, f::DiffableFunction)
    print(io, typeof(f))
end

function Base.show(io::IO, ::MIME"text/plain", f::DiffableFunction)
    print(
        io,
        """
$(typeof(f))
    inputs: $(f.grad_field.inputs)
    output: $(f.grad_field.output)
    generation: $(f.grad_field.generation)"""
    )
end

