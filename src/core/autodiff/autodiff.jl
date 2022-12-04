module AutoDiff

abstract type Variable end
abstract type DiffableFunction  end

include("device.jl")
include("variable.jl")
include("function.jl")
include("propagation.jl")

export DiffableFunction, BinaryOperation, GradField, Variable, Scalar, AbstractTensor, Tensor, CuTensor, transport!, forward, backward!

end
