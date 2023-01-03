module JITrench

import Base

# begin AutoDiff module
include("core/autodiff/autodiff.jl")
# end

include("utils/error.jl")

include("core/operators.jl")
include("core/math_functions.jl")

include("arr/mapfunction.jl")
include("arr/reshape.jl")
include("arr/flatten.jl")
include("arr/transpose.jl")
include("arr/sum.jl")
include("arr/sum_to.jl")
include("arr/broadcast_to.jl")
include("arr/matmul.jl")
include("arr/getindex.jl")

include("utils/out.jl")
include("utils/plot.jl")
include("utils/utils.jl")


export Variable,
    DiffableFunction,
    backward!,
    parameters,
    cleargrad!,
    flatten,
    matmul,
    sum_to,
    sigmoid,
    mean_squared_error,
    linear,
    Model,
    Layer,
    SGD,
    @diff!

end
