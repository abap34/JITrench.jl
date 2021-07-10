import Base

abstract type Layer end



function (layer::Layer)(x...)
    outputs = forward(layer, x...)
    return outputs
end
    

mutable struct Parameters
    _dict
    Parameters() = new(Dict{Symbol, Variable}())
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

