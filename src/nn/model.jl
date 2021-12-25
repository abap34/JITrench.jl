""" 
    Model
Abstract type representing a model.

Child types of this abstract type are recognized as models by JITrench.

For more information, refer to the "Neural Networks" section of the documentation (here).
"""
abstract type Model end

"""
    plot_model(model::Model, inputs...,  to_file="model.png")
plot model by graphviz.
Note that JITrench builds the graph at the time of calculation, so some kind of input is required.
"""
function plot_model(model::Model, inputs...; to_file="model.png")
    y = model(inputs...)
    plot_graph(y, to_file=to_file)
end

# is_layer_def(ex) = (ex isa Expr) && (ex.head != :(=)) && (eval(ex.args[2]) <: Layer)

# get_layer_field_from_def(ex) = (def -> def.args[1]).(filter(is_layer_def, ex.args[3].args))

# layers(model::Model) = throw(NotImplemetedError("layers(::$(typeof(model))) is not implemented. Use @model or implement it directly."))


# macro model(ex)
#     layer_fields = get_layer_field_from_def(ex)
#     struct_name = ex.args[2].args[1]
#     quote
#         $ex
#         JITrench.layers(model::$(struct_name)) = getproperty.(Ref(model), $layer_fields)
#     end |> esc
# end

#TODO: better impl
function parameters(model::Model)
    params = []
    for layer in layers(model)
        for param in parameters(layer)
            push!(params, param)
        end
    end
    return params
end

"""
    cleargrads!(model::Model; skip_uninitialized=false)

clear information about grad in all parameters in `model`.
if skip_uninitialized is true, parameters which is uninitialized(i.e nothing) is skipped.
"""
function cleargrads!(model::Model; skip_uninitialized=false)
    for layer in layers(model)
        for param in parameters(layer)
            if param isa Nothing
                if skip_uninitialized
                    continue
                else
                    throw(DomainError("It is not possible to operate to clear uninitialized parameters."))    
                end
            else
                param.grad = nothing
            end
        end
    end
end

