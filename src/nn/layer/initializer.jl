import Base
using ..JITrench: Device, CPU, GPU

struct Initializer
    weight :: OrderedDict{String, Dict{String, <:AbstractTensor}}
    current_shape :: Vector{Int}
    name_controller :: DefaultDict{DataType, Int}
    device :: Device
    function Initializer(in_shape::Tuple; device::Device=CPU()) 
        in_shape_vec = Vector{Int}(undef, length(in_shape))
        for (i, s) in enumerate(in_shape)
            if s isa Nothing
                in_shape_vec[i] = -1
            else
                in_shape_vec[i] = s
            end
        end
        return new(OrderedDict{String, Dict{String, <:AbstractTensor}}(), in_shape_vec, DefaultDict{DataType, Int}(0), device)
    end
end

function init(model, initializer::Initializer)
    model(initializer)
    return initializer.weight
end

function Base.broadcasted(f, initializer::Initializer)
    f(initializer)
end

function JITrench.call!(F::Type{<:DiffableFunction}, initializer::Initializer) 
    register!(
        initializer,
        F,
        Dict{String, Tensor}()
    )
    return initializer    
end

function register!(parameters::Initializer, layer_type::Type{<:Layer}, weight_dict::Dict{String, <:AbstractTensor}) 
    parameters.name_controller[layer_type] += 1
    parameters.weight[string(layer_type) * string(parameters.name_controller[layer_type])] = weight_dict
end

function register!(parameters::Initializer, func_type::Type{<:DiffableFunction}, weight_dict::Dict{String, <:AbstractTensor}) 
    parameters.name_controller[func_type] += 1
    parameters.weight[string(func_type) * string(parameters.name_controller[func_type])] = weight_dict
end


