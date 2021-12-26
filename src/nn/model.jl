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
    cleargrads!(model::Model; skip_uninit=false)

clear information about grad of the layers which is returned from `layers`.
if skip_uninit is true, parameters which is uninit(i.e nothing) is skipped.
"""
function cleargrads!(model::T, layers::Function; skip_uninit=false) where T <: Model
    for layer in layers(model)
        for param in parameters(layer)
            if param isa Nothing
                if skip_uninit
                    continue
                else
                    throw(DomainError("It is not possible to operate to clear uninit parameters."))    
                end
            else
                param.grad = nothing
            end
        end
    end
end


