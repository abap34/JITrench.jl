module AutoDiff

abstract type JTFunction end
abstract type Variable end
abstract type DiffableFunction  <: JTFunction end


include("device.jl")
include("variable.jl")
include("function_utils.jl")
include("function.jl")
include("broadcast/broadcast.jl")
include("propagation.jl")
include("broadcast/sum_to.jl")
include("broadcast/broadcast_to.jl")

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
