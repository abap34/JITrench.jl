abstract type Model end

function plot(model::Model, inputs...; to_file="model.png")
    y = forward(model, inputs...)
    plot_graph(y, to_file=to_file)
end

is_layer_def(ex) = (ex isa Expr) && (ex.head != :(=)) && (eval(ex.args[2]) <: Layer)

get_layer_field_from_def(ex) = (def -> def.args[1]).(filter(is_layer_def, ex.args[3].args))

macro model(ex)
    layer_fields = get_layer_field_from_def(ex)
    struct_name = ex.args[2].args[1]
    quote
        $ex
        layers(model::$(struct_name)) = getproperty.(Ref(model), $layer_fields)
    end |> esc
end

function cleargrads!(model::Model; skip_uninitialized=false)
    for layer in Main.layers(model)
        for (_, param) in parameters(layer)
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