using JITrench


"""
    @nograd

Disable gradient calculation in the scope of the macro.

# Example
```julia
julia> x = Tensor([1, 2, 3])
3-element Tensor{Vector{Int64}}:
 1
 2
 3

julia> y = 2x
3-element Tensor{Vector{Int64}}:
 2
 4
 6

julia> JITrench.AutoDiff.@nograd y = 2x
3-element Vector{Int64}:
 2
 4
 6

julia> JITrench.AutoDiff.@nograd begin
           y = 2x
       end
3-element Vector{Int64}:
 2
 4
 6
```
"""
macro nograd(ex)
    JITrench.nograd = true
    res = Core.eval(__module__, ex)
    JITrench.nograd = false
    return res
end

