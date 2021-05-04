import Base: Broadcast

abstract type Functional <: TrenchObject end

mutable struct GradField
    inputs
    outputs
    generation
    GradField(inputs=nothing,  outputs=nothing, generation=nothing) = new(inputs, outputs, generation)
end

