import Base

mutable struct Broadcasting <: DiffableFunction
    f
    true_func
    grad_field :: GradField
    Broadcasting(f, true_func) = new(f, true_func, GradField()) 
end




_func_to_jt_struct = Dict(
    (+) => Add, 
    (-) => Sub, 
    (*) => Mul,
    (/) => Div,
    (^) => Pow,
    (sin) => Sin,
    (cos) => Cos,
    (tan) => Tan,
    (log) => Log,
)


for (func, jt_func) in _func_to_jt_struct
    Base.broadcasted(::typeof(func), x::Variable...) = Broadcasting(func, jt_func(GradField()))(x...)
    Base.broadcasted(::typeof(func), x1::Variable, x2) = Broadcasting(func, jt_func(GradField()))(x1, Variable(x2))
    Base.broadcasted(::typeof(func), x1, x2::Variable) = Broadcasting(func, jt_func(GradField()))(Variable(x1), x2)
end

function forward(f::Broadcasting, x1...)
    return Base.materialize(Base.broadcasted(f.f, x1...))
end


function backward(f::Broadcasting, gy)
    f.true_func.grad_field = f.grad_field
    if length(f.grad_field.inputs) ==  1
        gx = backward(f.true_func, gy)
        return gx
    else 
        x1 = f.grad_field.inputs[1]
        x2 = f.grad_field.inputs[2]
        gx1, gx2 = backward(f.true_func, gy)
        if size(x1.values) != size(x2.values)
            # println("broadcast!")
            # println("gx1 used sum_to to change $(size(gx1.values)) -> $(size(x1.values))")
            # println("gx2 used sum_to to change $(size(gx2.values)) -> $(size(x2.values))")
            gx1 = sum_to(gx1, size(x1.values))
            gx2 = sum_to(gx2, size(x2.values))
        end
        return gx1, gx2
    end
end