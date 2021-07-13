include("../funcitons/linear.jl")

mutable struct Linear <: Layer
    param :: Parameters
    in_dim
    out_dim
    initialized
    initial_method 
    function Linear(out_dim; in_dim=nothing, initial_method="xavier")
        if in_dim isa Nothing
            layer = new(Parameters(), in_dim, out_dim, false, initial_method)
            layer.param.b = Variable(zeros(1, out_dim), name="b")
            return layer
        else
            layer = new(Parameters(), in_dim, out_dim, true, initial_method)
            layer.param.b = Variable(zeros(1, out_dim))
            if initial_method == "xavier"
                W = xavier(in_dim, out_dim)
            elseif initial_method == "he"
                W = he(in_dim, out_dim)
            else initial_method isa Function
                W = initial_method(in_dim, out_dim)
            end    
            layer.param.W = Variable(W, name="W")
            return layer
        end
    end
end



function forward(layer::Linear, x)
    if !(layer.initialized)
        layer.in_dim = size(x.values)[2]
        layer.initialized = true
        if layer.initial_method == "xavier"
            W = xavier(layer.in_dim, layer.out_dim)
        elseif layer.initial_method == "he"
            W = he(layer.in_dim, layer.out_dim)
        else layer.initial_method isa Function
            W = layer.initial_method(layer.in_dim, layer.out_dim)
        end   
        layer.param.W = Variable(W, name="W")
    end 
    return linear(x, layer.param.W, layer.param.b)
end

