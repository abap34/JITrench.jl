module JITrench

import Base

# begin AutoDiff module
include("core/autodiff/autodiff.jl")
# end
include("core/operators.jl")
include("core/math_functions.jl")

# begin ArrOperator 
include("core/arr/arr.jl")
# end


# begin NN
include("nn/nn.jl")
# end

include("utils/error.jl")
include("utils/out.jl")
include("utils/plot.jl")
include("utils/utils_macro.jl")



export AutoDiff, backward!, Scalar, Tensor, CuTensor

end