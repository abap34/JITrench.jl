module AutoDiff

abstract type Variable end
abstract type DiffableFunction end


include("device.jl")
include("variable.jl")
include("function_utils.jl")
include("function.jl")
include("propagation.jl")

const ScalarTypes = Union{Real, Scalar}
const TensorTypes = Union{AbstractArray, Tensor, CuTensor}

export DiffableFunction,
    BinaryOperator,
    UnaryOperator,
    GradField,
    AdditionalField,
    Variable,
    Scalar,
    ScalarTypes,
    TensorTypes,
    AbstractTensor,
    Tensor,
    CuTensor,
    transport!,
    forward,
    backward,
    backward!

end
