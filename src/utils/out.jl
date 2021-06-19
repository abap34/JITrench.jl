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
        output *= "creator: User-Defined (nothing)"
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




# print()



#=
Type: JITrench.Mul
GradField:
    input: $input
    output: $output
    generation: $generation
=#
function Base.show(io::IO, f::Functional)
    print(io, typeof(f))
end

# REPL
function Base.show(io::IO, ::MIME"text/plain", f::Functional) 
    print(io,"""
    Type: $(typeof(f))
      GradField:
        inputs: $(f.grad_field.inputs)
        outputs: $(f.grad_field.outputs)
        generation: $(f.grad_field.generation)""")
end




