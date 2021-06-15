get_values(var) = var isa Variable ? var.values : var


function get_gy(f)
    outputs = f.grad_field.outputs
    return [output.grad for output in outputs]
end

function ones_like(x)
    shape = size(x)
    return isempty(shape) ? 1 : ones(shape)
end

function cleargrad!(var::Variable)
    var.grad = nothing
end


function as_arr(x::T)::AbstractArray{T} where T <: Real 
    return [x]
end

function as_arr(x::AbstractArray)
    return x
end    
    
function as_tuple(x)
    return (x,)
end

function as_tuple(x::Tuple)
    return x
end


