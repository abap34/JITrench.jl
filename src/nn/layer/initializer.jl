using DataStructures: OrderedDict, DefaultDict
using ..JITrench 

mutable struct ParameterRegister{T <: Tuple}
    weight :: OrderedDict{String, Dict{String, <:Tensor}}
    current_shape :: T
    name_controller :: DefaultDict{DataType, Int}
    function ParameterRegister(in_shape::T) where {T}
        return new{T}(OrderedDict{String, Tensor}(), in_shape, DefaultDict{DataType, Int}(0))
    end
end

struct Initializer{T <: Tuple}
    in_shape :: T
    parameters :: ParameterRegister
    function Initializer(in_shape::T) where T
        return new{T}(in_shape, ParameterRegister(in_shape))
    end
end

function init(model, initializer::Initializer)
    model(initializer)
    return initializer.parameters
end


function Base.broadcasted(f, args::Tuple{AbstractTensor, ParameterRegister, Dict})
    args[1].req_broadcast = true
    return f(args)
    args[1].req_broadcast = false
end


function Base.broadcasted(f, initializer::Initializer)
    f(initializer)
end




function JITrench.call!(F::Type{<:DiffableFunction}, initializer::Initializer{<:Tuple}) 
    register!(
        initializer.parameters,
        F,
        Dict{String, Tensor}()
    )
    return initializer    
end

function JITrench.call!(F::Type{<:DiffableFunction}, args::Tuple{AbstractTensor, ParameterRegister, Dict})
    x, parameter, name_controller = args
    result =  JITrench.call!(F, x)
    return (
        result,
        parameter,
        name_controller
    )
end

function (layer::Layer)(args::Tuple{AbstractTensor, ParameterRegister, Dict})
    x, parameter, name_controller = args
    layer_type = typeof(layer)
    name_controller[layer_type] += 1
    key = string(layer_type) * string(name_controller[layer_type])
    weight = parameter.weight[key]
    result = apply(layer, weight, x)
    return (
        result,
        parameter,
        name_controller
    )
end

function register!(parameters::ParameterRegister, layer_type::Type{<:Layer}, weight_dict::Dict{String, <:Tensor}) 
    parameters.name_controller[layer_type] += 1
    parameters.weight[string(layer_type) * string(parameters.name_controller[layer_type])] = weight_dict
end

function register!(parameters::ParameterRegister, func_type::Type{<:DiffableFunction}, weight_dict::Dict{String, <:Tensor}) 
    parameters.name_controller[func_type] += 1
    parameters.weight[string(func_type) * string(parameters.name_controller[func_type])] = weight_dict
end

function apply(model::Function, x, param)
    name_controller = Dict{DataType, Int}()
    for key in keys(param.name_controller)
        name_controller[key] = 0
    end
    model((x, param, name_controller))
end

function result(arg::Tuple{AbstractTensor, ParameterRegister, Dict})
    return arg[1]
end

function result(arg)
    return arg
end


