module JITrench

nograd :: Bool = false

import Base

# begin AutoDiff module
include("core/autodiff/autodiff.jl")
# end


export AutoDiff, DiffableFunction, backward!, Scalar, AbstractTensor, Tensor, CuTensor

Device = AutoDiff.Device
GPU = AutoDiff.GPU
CPU = AutoDiff.CPU

include("core/functions.jl")

# begin ArrOperator 
include("core/arr/arr.jl")
# end


# begin NN
include("nn/nn.jl")
# end

include("utils/error.jl")
include("utils/plot.jl")
include("utils/utils_macro.jl")




end