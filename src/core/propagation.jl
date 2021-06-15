import DataStructures


function (f::Functional)(vars...)
    f_gradfield = f.grad_field
    f_gradfield.inputs = collect(vars)

    xs = get_values.(vars)
    ys = as_tuple(forward(f, xs...))
    
    f_gradfield.generation = minimum((x -> x.generation), f_gradfield.inputs)
    f_gradfield.outputs = [Variable(y, creator=f, grad=nothing, generation=f_gradfield.generation - 1) for y in ys]   
    
    return length(f_gradfield.outputs)  == 1 ? f_gradfield.outputs[1] : f_gradfield.outputs
end


function backward!(var::Variable)
    funcs = DataStructures.PriorityQueue{Functional,Int}()
    seen_set = Set{Functional}()

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
            x.grad = gx
            else
                x.grad = x.grad + gx
            end

            if (!(isnothing(x.creator))) && (!(x.creator in seen_set))
                push!(seen_set, x.creator)
                DataStructures.enqueue!(funcs, x.creator, x.creator.grad_field.generation)
            end
        
        end
    end
    return nothing
end

