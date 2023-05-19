module NN
    using ..JITrench
    include("function/functions.jl")
    include("layer/layer.jl")
    include("optimizer/optimizer.jl")
    include("data/data.jl")
    include("./utils.jl")
end