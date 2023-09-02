using ..JITrench 
using DataStructures: OrderedDict, DefaultDict

struct Parameter
    weight :: OrderedDict{String, Dict{String, <: AbstractTensor}}
    layer_names :: Vector{String}
    meta :: Dict{String, Any}
    function Parameter(weight::OrderedDict{String, Dict{String, <: AbstractTensor}})
        layer_names = Vector{String}(undef, length(weight))
        for (i, key) in enumerate(keys(weight))
            layer_names[i] = key
        end
        return new(weight, layer_names, Dict{String, Any}())
    end
end


abstract type Layer end

include("./init_distribution.jl")
include("./parameters.jl")
include("./initializer.jl")
include("./compute.jl")
include("./linear.jl")