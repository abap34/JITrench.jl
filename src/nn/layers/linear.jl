include("../funcitons/linear.jl")

mutable struct Linear <: Layer
    param :: Parameters
    in_dim
    out_dim
    lazy_init
    initial_method 
    function Linear(out_dim; in_dim=nothing, initial_method="xavier")
        if in_dim isa Nothing
            linear = new(Parameters(), in_dim, out_dim, true, initial_method)
            linear.param.b = Variable(zeros(1, out_dim))
            return linear
        else
            linear = new(Parameters(), in_dim, out_dim, false, initial_method)
            linear.param.b = Variable(zeros(1, out_dim))
            if initial_method == "xavier"
                W = xavier(in_dim, out_dim)
            elseif initial_method == "he"
                W = he(in_dim, out_dim)
            else initial_method isa Function
                W = initial_method(in_dim, out_dim)
            end    
            self.param.W = Variable(W)
            return linear
        end
    end
end



function forward(layer::Linear, x)
    if layer.lazy_init
        layer.in_dim = size(x.values)[2]
        layer.lazy_init = "done"
        if layer.initial_method == "xavier"
            W = xavier(layer.in_dim, layer.out_dim)
        elseif layer.initial_method == "he"
            W = he(layer.in_dim, layer.out_dim)
        else layer.initial_method isa Function
            W = initial_method(layer.in_dim, layer.out_dim)
        end   
        layer.param.W = Variable(W)
    end 
    return linear(x, layer.param.W, layer.param.b)
end

