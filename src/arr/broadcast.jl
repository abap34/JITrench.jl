import Base

mutable struct Broadcasting{F} <: DiffableFunction
    f :: F
    true_func
    grad_field :: GradField
end

Broadcasting(f, true_func) = Broadcasting(f, true_func, GradField()) 




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
    (exp) => Exp
)

# fix https://github.com/abap34/JITrench.jl/issues/11
Base.broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::Variable, ::Val{c}) where c = Broadcasting(^, Pow(GradField(), c))(x)
Base.broadcasted(::typeof(^), x::Variable, c) = Broadcasting(^, Pow(GradField(), c))(x)

# TODO: remove unnecessary def
for (func, jt_func) in _func_to_jt_struct
    (func == ^) && (continue)
    Base.broadcasted(::typeof(func), x::Variable...) = Broadcasting(func, jt_func(GradField()))(x...)
    Base.broadcasted(::typeof(func), x1::Variable, x2) = Broadcasting(func, jt_func(GradField()))(x1, Variable(x2))
    Base.broadcasted(::typeof(func), x1, x2::Variable) = Broadcasting(func, jt_func(GradField()))(Variable(x1), x2)
end

function forward(f::Broadcasting, x...)
    forward.(Ref(f.true_func), x...)
end

function forward(f::Broadcasting{typeof(^)}, x...)
    return Base.materialize(Base.broadcasted(f.f, x..., f.true_func.c))
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