using ..JITrench
import DataStructures

function forward(args...)
    throw(NotImplemetedError())
end

function backward(args...)
    throw(NotImplemetedError())
end

function out_to_tensor(
    y::Real,
    generation::Int;
    creator = nothing,
    grad = nothing,
    req_broadcast = false,
) 
    return Scalar(
        y,
        creator = creator,
        grad = grad,
        generation = generation + 1,
        req_broadcast = req_broadcast,
    )
end

function out_to_tensor(
    y::AbstractArray,
    generation::Int;
    creator = nothing,
    grad = nothing,
    req_broadcast = false,
)   return Tensor(
        y,
        creator = creator,
        grad = grad,
        generation = generation + 1,
        req_broadcast = req_broadcast,
    )
end

function out_to_tensor(
    y::CuArray,
    generation::Int,
    device_idx::Int;
    creator = nothing,
    grad = nothing,
    req_broadcast = false,
) return CuTensor(
        y,
        creator = creator,
        grad = grad,
        generation = generation + 1,
        req_broadcast = req_broadcast,
        device_idx = device_idx,
    )
end

@inline function make_func(
    ::Type{F},
    additional_field,
    y,
    inputs,
    gen,
) where {F <: DiffableFunction}
    gf = GradField(inputs, out_to_tensor(y, gen), gen)
    func = F(gf, additional_field)
    gf.output.creator = func
    return func
end


@inline function make_func(::Type{F}, y, inputs, gen) where {F <: DiffableFunction}
    gf = GradField(inputs, out_to_tensor(y, gen), gen)
    func = F(gf)
    gf.output.creator = func
    return func
end

@inline function make_func(
    ::Type{F},
    y::T,
    inputs,
    gen,
    device_idx,
) where {T <: CuArray, F <: DiffableFunction}
    gf = GradField(inputs, out_to_tensor(y, gen, device_idx), gen)
    func = F(gf)
    gf.output.creator = func
    return func
end

function call!(F::Type{<:UnaryOperator}, x::T; nograd = false) where {T <: Variable}
    if x.req_broadcast
        return call!(Broadcasting{F}, x)
    end
    inputs = (x,)
    y = forward(F, x.values)
    (nograd) && (return y)
    func = make_func(F, y, inputs, x.generation)
    return func.grad_field.output
end


function call!(
    F::Type{<:BinaryOperator},
    x1::Variable,
    x2::Variable;
    nograd = false,
) 
    inputs = (x1, x2)
    y = forward(F, x1.values, x2.values)
    (nograd) && (return y)
    gen = min(x1.generation, x2.generation)
    func = make_func(F, y, inputs, gen)
    return func.grad_field.output
end



function call!(
    F::Type{<:BinaryOperator},
    x1::CuTensor,
    x2::CuTensor;
    nograd = false,
) device_idx = check_same_device(x1.device, x2.device)
    inputs = (x1, x2)
    y = forward(F, x1.values, x2.values)
    (nograd) && (return y)
    gen = min(x1.generation, x2.generation)
    func = make_func(F, y, inputs, gen)
    return func.grad_field.output
end



function call!(
    F::Type{<:UnaryOperator},
    additional_field::AdditionalField,
    x::Variable,
    nograd = false,
) 
if x.req_broadcast
        return call!(Broadcasting{F}, additional_field, x)
    end
    inputs = (x,)
    y = forward(F, additional_field, x.values)
    (nograd) && (return y)
    func = make_func(F, additional_field, y, inputs, x.generation)
    return func.grad_field.output
end

function call!(
    F::Type{<:BinaryOperator},
    additional_field::AdditionalField,
    x1::Variable,
    x2::Variable;
    nograd = false,
) 
    inputs = (x1, x2)
    y = forward(F, additional_field, x1.values, x2.values)
    (nograd) && (return y)
    gen = min(x1.generation, x2.generation)
    func = make_func(F, additional_field, y, inputs, gen)
    return func.grad_field.output
end


function call!(
    F::Type{<:BinaryOperator},
    additional_field::AdditionalField,
    x1::CuTensor,
    x2::CuTensor;
    nograd = false,
)
    device_idx = check_same_device(x1.device, x2.device)
    inputs = (x1, x2)
    y = forward(F, additional_field, x1.values, x2.values)
    (nograd) && (return y)
    gen = min(x1.generation, x2.generation)
    func = make_func(F, additional_field, y, inputs, gen)
    return func.grad_field.output
end



@inline ones_like(x::CuTensor) =
    CuTensor(ones(eltype(x.values), size(x.values)), device_idx = x.device.idx)

@inline get_gy(f::DiffableFunction) = f.grad_field.output.grad

@inline function set_grad!(x::Variable, gx) 
    set_grad!(x, x.grad, gx)
end

@inline function set_grad!(x::Variable, ::Nothing, gx) 
    x.grad = gx
end

@inline function set_grad!(x::Variable, _, gx)
    x.grad = x.grad + gx
end

@inline function set_grad!(
    x::Variable,
    gx::Union{Real, AbstractArray};
    nograd::Bool,
) 
    if nograd
        set_grad!(x, x.grad, gx)
    end
end


@inline function set_grad!(x::T, gx::S; nograd=true::Bool) where {T <: Variable, S <: Variable}
    if nograd
        set_grad!(x, x.grad, gx.values)
    end
end

@inline function update_que!(
    f::DiffableFunction,
    seen_set::Set{DiffableFunction},
    pq::PriorityQueue{DiffableFunction, Int},
) 
    if !(f in seen_set)
        push!(seen_set, f)
        DataStructures.enqueue!(pq, f, f.grad_field.generation)
    end
end


@inline function update_que!(
    f::Nothing,
    seen_set::Set{DiffableFunction},
    pq::PriorityQueue{DiffableFunction, Int},
)
    # nothing to do
end

function backward!(y::Scalar; retain_grad = false, create_graph = false)
    que = DataStructures.PriorityQueue{DiffableFunction, Int}(Base.Order.Reverse)
    seen_set = Set{DiffableFunction}()
    if y.grad isa Nothing
        if create_graph
            y.grad = ones_like(y)
        else
            y.grad = ones_like(y.values)
        end
    end
    DataStructures.enqueue!(que, y.creator, 1)
    push!(seen_set, y.creator)
    while !(isempty(que))
        f = DataStructures.dequeue!(que)
        calculate_grad!(
            f,
            seen_set,
            que,
            retain_grad = retain_grad,
            create_graph = create_graph,
        )
    end
    return nothing
end

function calculate_grad!(
    f::BinaryOperator,
    seen_set::Set{DiffableFunction},
    que::PriorityQueue{DiffableFunction, Int};
    retain_grad = false,
    create_graph = false,
)
    gy = get_gy(f)
    gx1, gx2 = backward(f, gy)
    x1, x2 = f.grad_field.inputs
    if create_graph
        set_grad!(x1, gx1)
        set_grad!(x2, gx2)
    else
        set_grad!(x1, gx1, nograd = true)
        set_grad!(x2, gx2, nograd = true)
    end
    f1 = x1.creator
    f2 = x2.creator
    update_que!(f1, seen_set, que)
    update_que!(f2, seen_set, que)
    if !(retain_grad)
        f.grad_field.output.grad = nothing
    end
    return nothing
end


function calculate_grad!(
    f::UnaryOperator,
    seen_set::Set{DiffableFunction},
    que::PriorityQueue{DiffableFunction, Int};
    retain_grad = false,
    create_graph = false,
)
    gy = get_gy(f)
    gx = JITrench.backward(f, gy)
    x = f.grad_field.inputs[1]
    if create_graph
        set_grad!(x, gx)
    else
        set_grad!(x, gx, nograd = true)
    end
    f_c = x.creator
    update_que!(f_c, seen_set, que)
    if !(retain_grad)
        f.grad_field.output.grad = nothing
    end
    return nothing
end
