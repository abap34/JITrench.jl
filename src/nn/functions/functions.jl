module functions

using ...AutoDiff
import ...AutoDiff: forward, backward, call!


include("loss.jl")
include("linear.jl")


end