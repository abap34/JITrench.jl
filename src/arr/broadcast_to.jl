"""
    _broadcast_to(A, shape)
Apply broadcast to make `size(x.values)` as `shape`.
# Examples
```julia-repl
julia> _broadcast_to([1, 2, 3], (3, 3))
3×3 Matrix{Float64}:
 1.0  1.0  1.0
 2.0  2.0  2.0
 3.0  3.0  3.0

julia> _broadcast_to(1, (2, 2))
2×2 Matrix{Float64}:
 1.0  1.0
 1.0  1.0
```
"""
function _broadcast_to(A, shape)
    return zeros(shape) .+ A
end

mutable struct BroadcastTo <: DiffableFunction
    grad_field :: GradField
    shape
    x_shape
    BroadcastTo(shape, x_shape) = new(GradField(), shape, x_shape)
end

function forward(f::BroadcastTo, x)
    y = _broadcast_to(x, f.shape)
    return y
end

function backward(f::BroadcastTo, gy)
    gx = _sum_to(gy, f.x_shape)   
    return gx
end

"""
    broadcast_to(x::Variable, shape)

Apply broadcast to make `size(x.values)` as `shape`.　

# Examples
```julia-repl
julia> x = Variable([1, 2, 3])
name: nothing 
values: [1, 2, 3]
creator: User-Defined(nothing)

julia> JITrench.broadcast_to(x, (3, 2))
name: nothing 
values: [1.0 1.0; 2.0 2.0; 3.0 3.0]
creator: JITrench.BroadcastTo
```
"""
function broadcast_to(x::Variable, shape)
    if size(x.values) == shape
        return x
    else
        return BroadcastTo(shape, size(x.values))(x)
    end
end