module AutoDiff

abstract type Variable end
abstract type DiffableFunction end

include("device.jl")
include("variable.jl")
include("function_utils.jl")
include("function.jl")
include("propagation.jl")

export DiffableFunction,
    BinaryOperator,
    UnaryOperator,
    GradField,
    AdditionalField,
    Variable,
    Scalar,
    AbstractTensor,
    Tensor,
    CuTensor,
    transport!,
    forward,
    backward,
    backward!

end
