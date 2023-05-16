struct ComputeContext{T <: AbstractTensor}
    input :: T
    params :: Parameter
    name_controller :: DefaultDict{DataType, Int}
end


function Base.broadcasted(f, ctx::ComputeContext)
    ctx.input.req_broadcast = true
    y = f(args)
    ctx.input.req_broadcast = false
    return y
end


function JITrench.call!(F::Type{<:DiffableFunction}, ctx::ComputeContext)
    ctx.input = JITrench.call!(F, ctx.input)
    return ctx
end

function (layer::Layer)(ctx::ComputeContext)
    layer_type = typeof(layer)
    ctx.name_controller[layer_type] += 1
    key = string(layer_type) * string(name_controller[layer_type])
    weight = parameter.weight[key]
    ctx.input = apply(layer, weight, ctx.input)
    return ctx
end

function apply(model::Function, x::AbstractTensor, param::Parameter)
    name_controller = Dict{DataType, Int}()
    for key in keys(param)
        name_controller[key] = 0
    end
    model(ComputeContext(x, param, name_controller))
end

function result(ctx::ComputeContext)
    return ctx.input
end

function result(args...)
    return args
end