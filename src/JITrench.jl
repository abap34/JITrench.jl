module JITrench

import Base

"""
    TrenchObject
An abstract type that is the root of an object as implemented in 
Specifically, see also `subtypes(foo).`

```julia-repl
julia> subtypes(TrenchObject)
2-element Vector{Any}:
 DiffableFunction
 Variable
```
"""
abstract type TrenchObject end

"""
    DiffableFunction
An abstract type that is the parent type of differentiable functions;

all function in JITrench must be children of this type.

# Examples

```julia-repl
julia> subtypes(DiffableFunction)
24-element Vector{Any}:
 JITrench.Add
 JITrench.BroadcastTo
 JITrench.Broadcasting
 JITrench.Cos
 JITrench.Div
 JITrench.Exp
 JITrench.Flatten
 JITrench.GetIndex
 JITrench.GetIndexGrad
 JITrench.Log
 ⋮
 JITrench.Reshape
 JITrench.Sigmoid
 JITrench.Sin
 JITrench.Sub
 JITrench.Sum
 JITrench.SumTo
 JITrench.Tan
 JITrench.Transpose
 JITrench._Linear
```
"""
abstract type DiffableFunction  <: TrenchObject end


include("utils/error.jl")

include("core/variable.jl")
include("core/functions_utils.jl")
include("core/function.jl")
include("core/operators.jl")
include("core/propagation.jl")
include("core/math_functions.jl")

include("arr/reshape.jl")
include("arr/flatten.jl")
include("arr/transpose.jl")
include("arr/sum.jl")
include("arr/sum_to.jl")
include("arr/broadcast_to.jl")
include("arr/broadcast.jl")
include("arr/matmul.jl")
include("arr/getindex.jl")

include("utils/out.jl")
include("utils/plot.jl")
include("utils/utils.jl")

include("nn/utils.jl")
include("nn/layers/layer.jl")
include("nn/layers/linear.jl")
include("nn/funcitons/activation/sigmoid.jl")
include("nn/funcitons/loss/mean_squared_error.jl")

include("nn/model.jl")

include("nn/optimizer/optimizer.jl")
include("nn/optimizer/sgd.jl")



export Variable, DiffableFunction, backward!, parameters, cleargrad!, flatten, matmul, sum_to, sigmoid, mean_squared_error, linear, Model, Layer, SGD, @diff!

end