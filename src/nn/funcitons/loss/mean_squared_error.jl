mutable struct MeanSquaredError <: DiffableFunction
    grad_field :: GradField
end

function forward(::MeanSquaredError, y_true, y_pred)
    diff  = y_true .- y_pred
    loss = Base.sum((x -> x^2), diff) / length(diff)
    return loss
end


function backward(f::MeanSquaredError, gy)
    y_true, y_pred = f.grad_field.inputs
    diff = y_true .- y_pred
    gy = broadcast_to(gy, size(diff.values))
    gx1 = gy .* diff .* (2. / length(diff.values))
    gx2 = -gx1
    return (gx1, gx2)
end

"""
    mean_squared_error(y_true, y_pred)

# Examples
```julia-repl
julia> y_true = Variable([10, 20, 30])
name: nothing 
values: [10, 20, 30]
creator: User-Defined(nothing)

julia> y_pred = Variable([12, 20, 25])
name: nothing 
values: [12, 20, 25]
creator: User-Defined(nothing)

julia> mean_squared_error(y_true, y_pred)
name: nothing 
values: 9.666666666666666
creator: JITrench.MeanSquaredError
```
"""
mean_squared_error(y_true, y_pred) = MeanSquaredError(GradField())(y_true, y_pred)


