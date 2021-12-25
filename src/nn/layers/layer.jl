import Base

abstract type Layer end

function (layer::Layer)(x...)
    outputs = forward(layer, x...)
    return outputs
end


function cleargrads!(layer::Layer)
    for param in parameters(layer)
        cleargrad!(param)
    end
end

