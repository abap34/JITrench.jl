import DataStructures


function (f::DiffableFunction)(vars...)
    f_gradfield = f.grad_field
    f_gradfield.inputs = collect(vars)

    xs = get_values.(vars)
    ys = as_tuple(forward(f, xs...))
    
    f_gradfield.generation = minimum((x -> x.generation), f_gradfield.inputs)
    f_gradfield.outputs = [Variable(y, creator=f, grad=nothing, generation=f_gradfield.generation - 1) for y in ys]   
    
    return length(f_gradfield.outputs)  == 1 ? f_gradfield.outputs[1] : f_gradfield.outputs
end


function backward!(var::Variable; retain_grad=false)
    funcs = DataStructures.PriorityQueue{DiffableFunction,Int}()
    seen_set = Set{DiffableFunction}()

    if var.grad isa Nothing
        var.grad = Variable(ones_like(var.values))
    end
    
    DataStructures.enqueue!(funcs, var.creator, 1)
    push!(seen_set, var.creator)
    
    while !(isempty(funcs))
        f = DataStructures.dequeue!(funcs)
        gy = [output.grad for output in f.grad_field.outputs]

        gxs = backward(f, gy...)
        gxs = as_tuple(gxs)
        
        for (x, gx) in zip(f.grad_field.inputs, gxs)
            if x.grad isa Nothing
                if x.values isa Number
                    gx.values = gx.values[1]
                    x.grad = gx
                else
                    x.grad = gx
                end
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
