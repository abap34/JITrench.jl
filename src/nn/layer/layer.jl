using ..JITrench 

Parameter = OrderedDict{String, Dict{String, <: Tensor}}

abstract type Layer end

include("./init_distribution.jl")
include("./parameters.jl")
include("./initializer.jl")
include("./compute.jl")
include("./linear.jl")