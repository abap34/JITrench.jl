module functions

using ...AutoDiff
import ...AutoDiff: forward, backward, call!


include("loss.jl")
include("activation.jl")
include("linear.jl")


end