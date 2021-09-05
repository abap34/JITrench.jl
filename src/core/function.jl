"""
    GradField
A structure that contains all the necessary items for automatic differentiation.
All function types in JITrench have this type as a field named "grad_field".
"""
mutable struct GradField
    inputs :: Union{Nothing, Vector{Variable}}
    outputs :: Union{Nothing, Vector{Variable}}
    generation :: Union{Nothing, Int}
    function GradField(inputs=nothing, outputs=nothing, generation=nothing) 
        new(inputs, outputs, generation)    
    end
end

function Base.show(io::IO, f::DiffableFunction)
    print(io, typeof(f))
end

function Base.show(io::IO, ::MIME"text/plain", f::DiffableFunction) 
    print(io,"""
    Type: $(typeof(f))
      GradField:
        inputs: $(f.grad_field.inputs)
        outputs: $(f.grad_field.outputs)
        generation: $(f.grad_field.generation)""")
end