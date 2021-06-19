import Base

mutable struct Broadcasting <: Functional
    f
    true_func
    grad_field::GradField
    Broadcasting(f, true_func) = new(f, true_func, GradField()) 
end

Base.broadcasted(f, x::Variable...) = Broadcasting(f, nothing)(x...)
Base.broadcasted(f, x1::Variable, x2) = Broadcasting(f, nothing)(x1, Variable(x2))
Base.broadcasted(f, x1, x2::Variable) = Broadcasting(f, nothing)(Variable(x1), x2)




_op_to_jt_struct = Dict(
    (+) => :Add, 
    (-) => :Sub, 
    (*) => :Mul,
    (/) => :Div,
    (^) => :Pow
)

function forward(f::Broadcasting, x1...)
    return Base.materialize(Base.broadcasted(f.f, x1...))
end


function backward(f::Broadcasting, gy)
    if isnothing(f.true_func)
        if f.f == ^
            f.true_func = Pow(GradField(), f.grad_field.inputs[2])
            f.true_func.grad_field.inputs = as_arr(f.grad_field.inputs[1])
        else
            f.true_func = eval(_op_to_jt_struct[f.f])(GradField())
            f.true_func.grad_field.inputs = f.grad_field.inputs
        end
    end
    if f.f != ^ 
        x1 = f.grad_field.inputs[1]
        x2 = f.grad_field.inputs[2]
        gx1, gx2 = backward(f.true_func, gy)
        if size(x1.values) != size(x2.values)
            # println("broadcast!")
            # println("gx1 used sum_to to change $(size(gx1.values)) -> $(size(x1.values))")
            # println("gx2 used sum_to to change $(size(gx2.values)) -> $(size(x2.values))")
            # gx1 = sum_to(gx1, size(x1.values))
            gx2 = sum_to(gx2, size(x2.values))
        end
        return gx1, gx2
    else
        gx1 = backward(f.true_func, gy)
        return gx1
    end
end