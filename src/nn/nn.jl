module NN
    using ..JITrench
    include("function/functions.jl")
    include("layer/layer.jl")
    include("optimizer/optimizer.jl")
    include("./utils.jl")
end