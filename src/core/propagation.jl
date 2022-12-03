import DataStructures



function out_to_tensor(y::T, f::DiffableFunction, generation::Int; device="cpu", idx=0) where T <: AbstractArray
    if device == "cpu"
        return [Tensor(y, creator=f, grad=nothing, generation=generation + 1, req_broadcast=false), ]
    else
        return [CuTensor(y, creator=f, grad=nothing, generation=generation + 1, req_broadcast=false, device_idx=idx), ]
    end
end


function (f::DiffableFunction)(var1::T, var2::S) where {T <: Tensor, S <: Tensor}
    gf = f.grad_field
    gf.inputs = [var1, var2]

    xs = [var1.values, var2.values]
    ys = forward(f, var1.values, var2.values)

    gf.generation = min(var1.generation, var2.generation)
    gf.outputs = out_to_tensor(ys, f, gf.generation, device="cpu")

    return length(gf.outputs) == 1 ? gf.outputs[1] : gf.outputs
end


function (f::DiffableFunction)(var1::T, var2::S) where {T <: CuTensor, S <: CuTensor}
    if var1.device_idx != var2.device_idx 
        throw("$(var1.device_idx), $(var2.device_idx)")
    end
    gf = f.grad_field
    gf.inputs = [var1, var2]

    xs = [var1.values, var2.values]
    ys = forward(f, var1.values, var2.values)

    gf.generation = min(var1.generation, var2.generation)
    gf.outputs = out_to_tensor(ys, f, gf.generation, device="gpu", idx=var1.device_idx)

    return length(gf.outputs) == 1 ? gf.outputs[1] : gf.outputs
end
    

# function (f::SingleReturnFunction)(vars...)
#     f_gradfield = f.grad_field
#     f_gradfield.inputs = collect(vars)
#     for var in vars
#         if var.req_broadcast
#             f = Broadcasting(pure_func(f), f)
#             y = f(vars...)
#             y.req_broadcast = true
#             return y
#         end
#     end
#     xs = get_values.(vars)
#     y = forward(f, xs...)
#     f_gradfield.generation = minimum((x -> x.generation), f_gradfield.inputs)
#     out = Variable(y, creator=f, grad=nothing, generation=f_gradfield.generation + 1, req_broadcast=false)
#     f_gradfield.outputs = [out]
#     return f_gradfield.outputs[1]
# end

"""
    backward(y::Variable; retain_grad=false)
Compute the back propagation, `y` as the end of the computational graph.
For a more detailed explanation, see the documentation (here).

```julia-repl
julia> x
name: nothing 
values: 1
creator: User-Defined(nothing)

julia> x = Variable(1)
name: nothing 
values: 1
creator: User-Defined(nothing)

julia> y = x +  2
name: nothing 
values: 3
creator: JITrench.Add

julia> backward!(y)

julia> x.grad
name: nothing 
values: 1
creator: User-Defined(nothing)
```
"""
function backward!(y::Variable; retain_grad=false)
    funcs = DataStructures.PriorityQueue{DiffableFunction,Int}(Base.Order.Reverse)
    seen_set = Set{DiffableFunction}()
    if y.grad isa Nothing
        y.grad = Variable(ones_like(y.values))
    end
    DataStructures.enqueue!(funcs, y.creator, 1)
    push!(seen_set, y.creator)
    while !(isempty(funcs))
        f = DataStructures.dequeue!(funcs)
        if f isa SingleReturnFunction
            gy = [f.grad_field.outputs[1].grad]
        end
        gy = [output.grad for output in f.grad_field.outputs]
        gxs = as_tuple(backward(f, gy...))
        for (x, gx) in zip(f.grad_field.inputs, gxs)
            if x.grad isa Nothing
                x.grad = gx
            else
                x.grad = x.grad + gx
            end
            if (!(isnothing(x.creator))) && (!(x.creator in seen_set))
                push!(seen_set, x.creator)
                DataStructures.enqueue!(funcs, x.creator, x.creator.grad_field.generation)
            end
        end
        if !(retain_grad)
            for y in f.grad_field.outputs
                y.grad = nothing
            end
        end
    end
    return nothing
end
