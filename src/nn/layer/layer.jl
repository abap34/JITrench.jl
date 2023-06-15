using ..JITrench 
using DataStructures: OrderedDict, DefaultDict

Parameter = OrderedDict{String, Dict{String, <: AbstractTensor}}

abstract type Layer end

include("./init_distribution.jl")
include("./parameters.jl")
include("./initializer.jl")
include("./compute.jl")
include("./linear.jl")