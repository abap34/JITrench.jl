import Base
using .AutoDiff
import .AutoDiff: call!, out_to_tensor

mutable struct Broadcasting{F} <: DiffableFunction
    grad_field :: GradField
end

function call!(F::Type{Broadcasting{TF}}, x::Tensor) where TF <: UnaryOperator
    inputs = (x, )
    y = forward.(TF, x.values)
    gen = x.generation
    gf = GradField(
        inputs,
        out_to_tensor(y, gen, req_broadcast=true),
        gen
    )
    func = Broadcasting{TF}(
        gf
    )
    gf.output.creator = func
    return gf.output
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
    (exp) => Exp
)

Base.broadcasted(::typeof(|>), x::T, f) where T <: AbstractTensor = Base.broadcasted(f, x)

# # fix https://github.com/abap34/JITrench.jl/issues/11
# Base.broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::Variable, ::Val{c}) where c = Broadcasting(^, Pow(GradField(), c))(x)
# Base.broadcasted(::typeof(^), x::Variable, c) = Broadcasting(^, Pow(GradField(), c))(x)

for (func, jt_func) in _func_to_jt_struct
    (func == ^) && (continue)
    Base.broadcasted(::typeof(func), x::T) where T <: AbstractTensor = call!(Broadcasting{jt_func}, x)
end



function forward(f::Broadcasting, x...)
    Base.materialize(Base.broadcasted(f.f, x...))
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
            gx1 = sum_to(gx1, size(x1.values))
            gx2 = sum_to(gx2, size(x2.values))
        end
        return gx1, gx2
    end
end


function Base.broadcasted(f::Function, x::T) where T <: AbstractTensor
    x.req_broadcast = true
    y = f(x)
    y.req_broadcast = false
    return y
end
