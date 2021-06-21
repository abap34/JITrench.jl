abstract type DiffableFunction <: TrenchObject end

mutable struct GradField
    inputs
    outputs
    generation
    GradField(inputs=nothing,  outputs=nothing, generation=nothing) = new(inputs, outputs, generation)
end

