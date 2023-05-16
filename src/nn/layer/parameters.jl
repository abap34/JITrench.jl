function iterate_layer(params::Parameter)
    return params
end

function iterate_all(params::Parameter)
    return Base.Iterators.map(x -> x.second, Iterators.flatten(values(params)))
end

function cleargrads!(params::Parameter)
    for param in iterate_all(params)
        JITrench.AutoDiff.cleargrad!(param)
    end
end




