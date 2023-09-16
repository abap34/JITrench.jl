using DataStructures
using GPUArraysCore

using ..JITrench

"""
    forward(F::Type{DiffableFunction}, args...)

Definition of forward calculation of F.
This function must be implemented for each `DiffableFunction`.
This function is called by `call!` function with `values` of inputs.

# Arguments
- `F::Type{DiffableFunction}` : Type of `DiffableFunction` which is called.
- `args...` : `values` of inputs.

# Returns
- `y` : `values` of output.

# Example
```julia-repl
julia>julia> JITrench.AutoDiff.forward(JITrench.Add, 1, 2)
3
```
"""
function forward(F::Type{DiffableFunction}, args...)
    throw(ArgumentError("Not Implemented forward function. args: $args"))
end

"""
    backward(f::DiffableFunction, gy)

Definition of backward calculation of F.
This function must be implemented for each `DiffableFunction`.

# Arguments
- `F::DiffableFunction` : Instance of `DiffableFunction` which is called.
- `gy` : propageted gradient from output.

# Returns
- `gxs...` : propageted gradient to inputs.

# Example
```julia-repl
julia>julia> JITrench.AutoDiff.backward(f, 1)
(1, 1)
```
"""
function backward(f::DiffableFunction, gy)
    throw(ArgumentError("Not Implemented backward function. args: $args"))
end



"""
    out_to_tensor(y, generation; creator=nothing, grad=nothing, req_broadcast=false)

Convert output of `forward` function to appropriate `Variable.` (e.g. `Scalar`, `Tensor`, `CuTensor`)

# Arguments
- `y` : `values` of output.
- `generation` : `generation` of output.
- `creator` : `creator` of output.
- `grad` : `grad` of output.
- `req_broadcast` : `req_broadcast` of output.

# Returns
- `out` : `Variable` which is converted from `y`.   
"""
function out_to_tensor end


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


"""
    make_func(::Type{F}, y, inputs, gen)

Create `DiffableFunction` instance from `y`, `inputs` and `gen`.
"""
function make_func end


"""
    make_func(::Type{F}, additional_field, y, inputs, gen)

Create `DiffableFunction` instance from `additional_field`, `y`, `inputs` and `gen`.

# Arguments
- `F::Type{F}` : Type of `DiffableFunction` which is created.
- `additional_field` : `additional_field` for `F`.
- `y` : `values` of output.
- `inputs` : `inputs` of `F`.
- `gen` : `generation` of output.
"""
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

"""
    make_func(::Type{F}, y, inputs, gen)

Create `DiffableFunction` instance from  `y`, `inputs` and `gen`.

# Arguments
- `F::Type{F}` : Type of `DiffableFunction` which is created.
- `y` : `values` of output.
- `inputs` : `inputs` of `F`.
- `gen` : `generation` of output.
"""
@inline function make_func(::Type{F}, y, inputs, gen) where {F <: DiffableFunction}
    gf = GradField(inputs, out_to_tensor(y, gen), gen)
    func = F(gf)
    gf.output.creator = func
    return func
end



"""
    make_func(::Type{F}, y::CuArray, inputs, gen, device_idx)

Create `DiffableFunction` instance from  `y`,  `inputs` , `gen` and `device_idx`.

# Arguments
- `F::Type{F}` : Type of `DiffableFunction` which is created.
- `y` : `values` of output which is `CuArray`.
- `inputs` : `inputs` of `F`.
- `gen` : `generation` of output.
"""
@inline function make_func(
    ::Type{F},
    y::T,
    inputs,
    gen,
    device_idx,
) where {T <: CuArray, F <: DiffableFunction}
    gf = GradField(inputs, out_to_tensor(y, gen, device_idx=device_idx), gen)
    func = F(gf)
    gf.output.creator = func
    return func
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


"""
    call!(F::Type{<:DiffableFunction}, xs...::Variable)

Call JITrench function with `xs...` as inputs.
"""
function call! end


"""
    call!(F::Type{<:DiffableFunction}, xs...::Variable)

Call JITrench function with `xs...` as inputs.

# Arguments
- `F::Type{<:DiffableFunction}` : Type of `DiffableFunction` which is called.
- `xs...::Variable` : `Variable` which is passed to `F`.

# Returns
- `y` : `values` of output.

# Example
```julia-repl
julia> x = Scalar(3)
Scalar{Int64}(3)

julia> y = Scalar(4)
Scalar{Int64}(4)

julia> JITrench.AutoDiff.call!(JITrench.Add, x, y)
Scalar{Int64}(7)
```
"""
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



"""
    call!(F::Type{<:DiffableFunction}, additional_field::AdditionalField, xs...::Variable)

Call JITrench function with `xs...` as inputs, and `additional_field` as optional argument.

# Arguments
- `F::Type{<:DiffableFunction}` : Type of `DiffableFunction` which is called.
- `additional_field::AdditionalField` : `AdditionalField` which is passed to `F`.
- `xs...::Variable` : `Variable` which is passed to `F`.

# Returns
- `y` : `values` of output.

# Example
```julia-repl
julia> x = Scalar(3)
Scalar{Int64}(3)

julia> y = JITrench.AutoDiff.call!(JITrench.Pow, JITrench.PowField(3), x)
Scalar{Int64}(27)

julia> y = JITrench.AutoDiff.call!(JITrench.Pow, JITrench.PowField(2), x)
Scalar{Int64}(9)
```
"""
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


"""
    backward(y::Scalar; retain_grad = false, create_graph = false)

Clculate the gradient for y for all inputs.
If `retain_grad` is `true`, all intermediate gradients are retained.
If `create_graph` is true, the graph of the backward pass is constructed, allowing to compute higher order derivative products.

# Arguments
- `y` : `Scalar` which is the output of the function. Gradient can be immplicitly created only for `Scalar`.
- `retain_grad` : If `true`, all intermediate gradients are retained.
- `create_graph` : If `true`, the graph of the backward pass is constructed, allowing to compute higher order derivative products.

# Returns
- `nothing`: This function returns nothing. The gradient for each input is stored in `grad` field of each input.

# Example
```julia-repl
julia> x = Scalar(3)
Scalar{Int64}(3)

julia> y = Scalar(2.0)
Scalar{Float64}(2.0)

julia> z = x * 2y
Scalar{Float64}(12.0)

julia> backward!(z)

julia> x.grad
4.0

julia> y.grad
6.0
```
"""
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

