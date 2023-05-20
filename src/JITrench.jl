module JITrench

nograd :: Bool = false

import Base

# begin AutoDiff module
include("core/autodiff/autodiff.jl")
# end


export AutoDiff, DiffableFunction, backward!, Scalar, AbstractTensor, Tensor, CuTensor


include("core/functions.jl")

# begin ArrOperator 
include("core/arr/arr.jl")
# end


# begin NN
include("nn/nn.jl")
# end

include("utils/error.jl")
include("utils/out.jl")
include("utils/plot.jl")
include("utils/util_functions.jl")
include("utils/utils_macro.jl")




end