module JITrench


include("utils/error.jl")


include("core/variable.jl")
include("core/functions_utils.jl")
include("core/function.jl")
include("core/operators.jl")
include("core/propagation.jl")
include("core/math_functions.jl")

include("arr/reshape.jl")
include("arr/flatten.jl")
include("arr/transpose.jl")
include("arr/sum.jl")
include("arr/sum_to.jl")
include("arr/broadcast_to.jl")
include("arr/broadcast.jl")
include("arr/matmul.jl")
include("arr/getindex.jl")

include("utils/out.jl")
include("utils/plot.jl")

include("nn/utils.jl")
include("nn/layers/layer.jl")
include("nn/layers/linear.jl")
include("nn/funcitons/activation/sigmoid.jl")
include("nn/funcitons/loss/mean_squared_error.jl")

include("nn/model.jl")

include("nn/optimizer/optimizer.jl")
include("nn/optimizer/sgd.jl")



export Variable, DiffableFunction, backward!, parameters, cleargrad!, flatten, matmul, sum_to, sigmoid, mean_squared_error, linear, Model, Layer, SGD

end