mutable struct ComputeContext{T <: AbstractTensor}
    input :: T
    params :: Parameter
    name_controller :: DefaultDict{String, Int}
end


function Base.broadcasted(f, ctx::ComputeContext)
    ctx.input.req_broadcast = true
    ctx.input = f(ctx.input)
    ctx.input.req_broadcast = false
    return ctx
end


function JITrench.call!(F::Type{<:DiffableFunction}, ctx::ComputeContext)
    ctx.input = JITrench.call!(F, ctx.input)
    return ctx
end

function (layer::Layer)(ctx::ComputeContext)
    layer_type = string(typeof(layer))
    ctx.name_controller[layer_type] += 1
    key = string(layer_type) * string(ctx.name_controller[layer_type])
    weight = ctx.params[key]
    ctx.input = apply(layer, weight, ctx.input)
    return ctx
end

function apply(model::Function, x::T, param::Parameter) where T
    name_controller = DefaultDict{String, Int}(0)
    for key in keys(param)
        name_controller[key] = 0
    end
    model(ComputeContext{T}(x, param, name_controller))
end

function result(ctx::ComputeContext)
    return ctx.input
end

function result(args...)
    return args
end