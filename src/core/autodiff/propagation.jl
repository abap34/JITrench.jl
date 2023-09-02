using DataStructures
using GPUArraysCore

using ..JITrench


function forward(args...)
    throw(ArgumentError("Not Implemented forward function. args: $args"))
end

function backward(args...)
    throw(ArgumentError("Not Implemented backward function. args: $args"))
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
)   
    return Tensor(
        y,
        creator = creator,
        grad = grad,
        generation = generation + 1,
        req_broadcast = req_broadcast,
    )
end

function out_to_tensor(
    y::GPUArraysCore.AbstractGPUArray,
    generation::Int;
    device_idx=0::Int,
    creator = nothing,
    grad = nothing,
    req_broadcast = false,
) 
    return CuTensor(
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

function call!(F::Type{<:DiffableFunction}, xs::Variable...)
    for x in xs
        if x.req_broadcast
            return call!(BroadcastWrapper{F}, x)
        end
    end
    inputs = xs
    xs_values = get_values.(xs)
    y = forward(F, xs_values...)
    (JITrench.nograd) && (return y)
    gen = minimum(x -> x.generation, xs)
    func = make_func(F, y, inputs, gen)
    return func.grad_field.output
end

function call!(F::Type{<:UnaryOperator}, x::Variable)
    if x.req_broadcast
        return call!(BroadcastWrapper{F}, x)
    end
    inputs = (x,)
    y = forward(F, x.values)
    (JITrench.nograd) && (return y)
    func = make_func(F, y, inputs, x.generation)
    return func.grad_field.output
end


function call!(
    F::Type{<:BinaryOperator},
    x1::Variable,
    x2::Variable;
) 
    if x1.req_broadcast || x2.req_broadcast
        return call!(BroadcastWrapper{F}, x1, x2)
    end
    inputs = (x1, x2)
    y = forward(F, x1.values, x2.values)
    if (JITrench.nograd)
        println("nograad!!!!!!")
        println("return:", y)
        return y
    end
    (JITrench.nograd) && (return y)
    gen = min(x1.generation, x2.generation)
    func = make_func(F, y, inputs, gen)
    return func.grad_field.output
end



function call!(
    F::Type{<:BinaryOperator},
    x1::CuTensor,
    x2::CuTensor;
) 
    if x1.req_broadcast || x2.req_broadcast
        return call!(BroadcastWrapper{F}, x1, x2)
    end
    device_idx = check_same_device(x1.device, x2.device)
    inputs = (x1, x2)
    y = forward(F, x1.values, x2.values)
    (JITrench.nograd) && (return y)
    gen = min(x1.generation, x2.generation)
    func = make_func(F, y, inputs, gen)
    return func.grad_field.output
end



function call!(
    F::Type{<:UnaryOperator},
    additional_field::AdditionalField,
    x::Variable,
) 
    if x.req_broadcast
        return call!(BroadcastWrapper{F}, additional_field, x)
    end
    inputs = (x,)
    y = forward(F, additional_field, x.values)
    (JITrench.nograd) && (return y)
    func = make_func(F, additional_field, y, inputs, x.generation)
    return func.grad_field.output
end

function call!(
    F::Type{<:BinaryOperator},
    additional_field::AdditionalField,
    x1::Variable,
    x2::Variable;
) 
    inputs = (x1, x2)
    y = forward(F, additional_field, x1.values, x2.values)
    (JITrench.nograd) && (return y)
    gen = min(x1.generation, x2.generation)
    func = make_func(F, additional_field, y, inputs, gen)
    return func.grad_field.output
end


function call!(
    F::Type{<:BinaryOperator},
    additional_field::AdditionalField,
    x1::CuTensor,
    x2::CuTensor;
)
    device_idx = check_same_device(x1.device, x2.device)
    inputs = (x1, x2)
    y = forward(F, additional_field, x1.values, x2.values)
    (JITrench.nograd) && (return y)
    gen = min(x1.generation, x2.generation)
    func = make_func(F, additional_field, y, inputs, gen)
    return func.grad_field.output
end



@inline ones_like(x::CuTensor) =
    CuTensor(ones(eltype(x.values), size(x.values)), device_idx = x.device.idx)

@inline get_gy(f::DiffableFunction) = f.grad_field.output.grad

function set_grad!(x::Variable, gx::Variable; nograd=true)
    if nograd
        gx = gx.values
        if x.grad isa Nothing
            x.grad = gx
        else
            x.grad = x.grad + gx
        end
    else
        if x.grad isa Nothing
            x.grad = gx
        else
            x.grad = x.grad + gx
        end
    end
end

function set_grad!(x::Variable, gx; nograd=true)
    if x.grad isa Nothing
        x.grad = gx
    else
        x.grad = x.grad + gx
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
    f::BroadcastWrapper,
    seen_set::Set{DiffableFunction},
    pq::PriorityQueue{DiffableFunction, Int},
) 
    if !(f in seen_set)
        push!(seen_set, f)
        DataStructures.enqueue!(pq, f, f.wrapped_func.grad_field.generation)
    end
end


@inline function update_que!(
    ::Nothing,
    ::Set{DiffableFunction},
    ::PriorityQueue{DiffableFunction, Int},
)
    # nothing to do
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
        set_grad!(x1, gx1, nograd=false)
        set_grad!(x2, gx2, nograd=false)
    else
        set_grad!(x1, gx1)
        set_grad!(x2, gx2)
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
    f::DiffableFunction,
    seen_set::Set{DiffableFunction},
    que::PriorityQueue{DiffableFunction, Int};
    retain_grad = false,
    create_graph = false,
)
    gy = get_gy(f)
    gxs = backward(f, gy)
    xs = f.grad_field.inputs
    nograd = !(create_graph)
    for i in eachindex(xs)
        set_grad!(xs[i], gxs[i], nograd=nograd)
        update_que!(xs[i].creator, seen_set, que)
    end        
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
    gx = backward(f, gy)
    x = f.grad_field.inputs[1]
    if create_graph
        set_grad!(x, gx, nograd=false)
    else
        set_grad!(x, gx)
    end
    f_c = x.creator
    update_que!(f_c, seen_set, que)
    if !(retain_grad)
        f.grad_field.output.grad = nothing
    end
    return nothing
end



function call!(f::Type{BroadcastWrapper{F}}, additional_field::AdditionalField, x::Variable, nograd=false) where F <: UnaryOperator
    y = forward.(Ref(F), Ref(additional_field), x.values)
    (nograd) && (return y)
    x.req_broadcast = false
    inputs = (x, )
    gf = GradField(
        inputs, 
        out_to_tensor(y, x.generation),
        x.generation
    )
    wrapped_func = F(gf, additional_field)
    func = BroadcastWrapper{F}(wrapped_func)
    wrapped_func.grad_field.output.creator = func
    return wrapped_func.grad_field.output
end




function call!(f::Type{BroadcastWrapper{F}}, additional_field::AdditionalField, x1::Variable, x2::Variable, nograd=false) where F <: BinaryOperator
    y = forward.(Ref(F), Ref(additional_field), x1.values, x2.values)
    (nograd) && (return y)
    x1.req_broadcast = false
    x2.req_broadcast = false
    inputs = (x1, x2)
    gen = min(x1.generation, x2.generation)
    gf = GradField(
        inputs, 
        out_to_tensor(y, gen),
        gen
    )
    wrapped_func = F(gf, additional_field)
    func = BroadcastWrapper{F}(wrapped_func)
    wrapped_func.grad_field.output.creator = func
    return wrapped_func.grad_field.output
end

function call!(f::Type{BroadcastWrapper{F}},  x::Variable) where F <: UnaryOperator
    y = forward.(Ref(F), x.values)
    (JITrench.nograd) && (return y)
    x.req_broadcast = false
    inputs = (x, )
    gf = GradField(
        inputs, 
        out_to_tensor(y, x.generation),
        x.generation
    )
    wrapped_func = F(gf)
    func = BroadcastWrapper{F}(wrapped_func)
    wrapped_func.grad_field.output.creator = func
    return wrapped_func.grad_field.output
end


function call!(f::Type{BroadcastWrapper{F}},  x1::Variable, x2::Variable) where F <: BinaryOperator
    y = forward.(Ref(F), x1.values, x2.values)
    (JITrench.nograd) && (return y)
    x1.req_broadcast = false
    x2.req_broadcast = false
    inputs = (x1, x2)
    gen = min(x1.generation, x2.generation)
    gf = GradField(
        inputs, 
        out_to_tensor(y, gen),
        gen
    )
    wrapped_func = F(gf)
    func = BroadcastWrapper{F}(wrapped_func)
    wrapped_func.grad_field.output.creator = func
    return wrapped_func.grad_field.output
end



function Base.broadcasted(f::Function, x::T) where T <: AbstractTensor
    x.req_broadcast = true
    y = f(x)
    y.req_broadcast = false
    return y
end


function Base.broadcasted(f::Function, x1::Variable, x2::Variable) 
    x1.req_broadcast = true
    x2.req_broadcast = true
    y = f(x1, x2)
    if !(JITrench.nograd)
        y.req_broadcast = false
    end
    return y
end


function Base.broadcasted(f::Function, x1::Variable, x2) 
    x1.req_broadcast = true
    y = f(x1, x2)
    if !(JITrench.nograd)
        y.req_broadcast = false
    end
    return y
end

function Base.broadcasted(f::Function, x1, x2::Variable) 
    x2.req_broadcast = true
    y = f(x1, x2)
    if !(JITrench.nograd)
        y.req_broadcast = false
    end
    return y
end
    

function calculate_grad!(
    wrapper::BroadcastWrapper{<:BinaryOperator},
    seen_set::Set{DiffableFunction},
    que::PriorityQueue{DiffableFunction, Int};
    retain_grad = false,
    create_graph = false,
)   
    gy = get_gy(wrapper.wrapped_func)
    gx1, gx2 = backward(wrapper.wrapped_func, gy)
    x1, x2 = wrapper.wrapped_func.grad_field.inputs
    if size(x1) != size(x2)
        _gx1 = sum_to(gx1, size(x1))
        _gx2 = sum_to(gx2, size(x2))
        if create_graph
            set_grad!(x1, _gx1, nograd=false)
            set_grad!(x2, _gx2, nograd=false)
        else
            set_grad!(x1, _gx1)
            set_grad!(x2, _gx2)
        end
    else
        if create_graph
            set_grad!(x1, _gx1, nograd=false)
            set_grad!(x2, _gx2, nograd=false)
        else
            set_grad!(x1, _gx1)
            set_grad!(x2, _gx2)
        end
    end
    f1 = x1.creator
    f2 = x2.creator
    update_que!(f1, seen_set, que)
    update_que!(f2, seen_set, que)
    if !(retain_grad)
        wrapper.wrapped_func.grad_field.output.grad = nothing
    end
    return nothing
end


function calculate_grad!(
    wrapper::BroadcastWrapper{<:UnaryOperator},
    seen_set::Set{DiffableFunction},
    que::PriorityQueue{DiffableFunction, Int};
    retain_grad = false,
    create_graph = false,
)   
    gy = get_gy(wrapper.wrapped_func)
    gx = backward(wrapper.wrapped_func, gy)
    x = wrapper.wrapped_func.grad_field.inputs[1]
    if create_graph
        set_grad!(x, gx, nograd=false)
    else
        set_grad!(x, gx)
    end
    f_c = x.creator
    update_que!(f_c, seen_set, que)
    if !(retain_grad)
        wrapper.wrapped_func.grad_field.output.grad = nothing
    end
    return nothing
end

function backward!(y::Scalar; retain_grad = false, create_graph = false)
    que = DataStructures.PriorityQueue{DiffableFunction, Int}(Base.Order.Reverse)
    seen_set = Set{DiffableFunction}()
    if y.grad isa Nothing
        if create_graph
            y.grad = Scalar(1.0)
        else
            y.grad = 1.0
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

function backward!(y::AbstractTensor; args...)
    throw(DomainError("Gradient can be immplicitly created only for `Scalar`"))
end

