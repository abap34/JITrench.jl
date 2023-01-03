module ArrOperator

using ..AutoDiff
import ..AutoDiff: forward, backward, call!


include("mapfunction.jl")
include("reshape.jl")
include("flatten.jl")
include("transpose.jl")
include("sum.jl")
include("sum_to.jl")
include("broadcast_to.jl")
include("matmul.jl")
include("getindex.jl")

end