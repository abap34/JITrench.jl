module JITrench

using Random

Random.seed!(10)

include("core/variable.jl")
include("core/functions_utils.jl")
include("core/function.jl")
include("core/operators.jl")
include("core/propagation.jl")
include("core/math_functions.jl")

include("arr/reshape.jl")
include("arr/transpose.jl")
include("arr/sum.jl")
include("arr/sum_to.jl")
include("arr/broadcast_to.jl")
include("arr/broadcast.jl")

include("utils/out.jl")
include("utils/plot.jl")
include("utils/util_functions.jl")

end