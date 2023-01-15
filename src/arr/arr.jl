module ArrOperator

using ..AutoDiff
import ..AutoDiff: forward, backward, call!


include("mapapply.jl")
include("reshape.jl")
include("flatten.jl")
include("transpose.jl")
include("sum.jl")
include("matmul.jl")
include("getindex.jl")

end