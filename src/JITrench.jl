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
include("arr/broadcast.jl")


include("utils/out.jl")
include("utils/plot.jl")

export Variable, backward!, cleargrad!

end
