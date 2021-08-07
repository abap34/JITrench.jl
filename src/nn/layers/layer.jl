import Base

abstract type Layer end

function (layer::Layer)(x...)
    outputs = forward(layer, x...)
    return outputs
end


function parameters(layer::Layer; value=true, key=false)
    if value && key
        return layer.param._dict
    elseif value && !(key)
        return values(layer.param._dict)
    elseif !(value) && key
        return keys(layer.param._dict)
    else
        throw(ArgumentError(""))
    end
end

function cleargrads!(layer::Layer)
    for param in parameters(layer)
        cleargrad!(param)
    end
end
mutable struct Parameters
    _dict
    Parameters() = new(Dict{Symbol, Union{Nothing, Variable}}())
end

function Base.setproperty!(param::Parameters, name::Symbol, value) 
    if name == (:_dict)
        throw(DomainError("`param._dict` is already used to contain values. Try another field name."))
    end
    param._dict[name] = value
end


function Base.getproperty(param::Parameters, name::Symbol)
    if name == (:_dict)
        return getfield(param, name)
    else
        return param._dict[name]
    end
end

