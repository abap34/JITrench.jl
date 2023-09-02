function iterate_layer(params::Parameter)
    return params.weight
end

function iterate_all(params::Parameter)
    return Base.Iterators.map(x -> x.second, Iterators.flatten(values(params.weight)))
end

function cleargrads!(params::Parameter)
    for param in iterate_all(params.weight)
        JITrench.AutoDiff.cleargrad!(param)
    end
end

function layer_names(params::Parameter)
    return params.layer_names
end



