include("../funcitons/layer_backend/linear.jl")

mutable struct Linear <: Layer
    W :: Union{Nothing, Variable}
    b :: Union{Nothing, Variable}
    in_dim :: Union{Nothing, Int}
    out_dim :: Int
    initial_method :: Union{String, Function}
    function Linear(out_dim; in_dim=nothing, initial_method="xavier")
        if in_dim isa Nothing
            b = Variable(zeros(1, out_dim), name="b")
            layer = new(nothing, b, in_dim, out_dim, initial_method)
            return layer
        else
            if initial_method == "xavier"
                W = xavier(in_dim, out_dim)
            elseif initial_method == "he"
                W = he(in_dim, out_dim)
            else initial_method isa Function
                W = initial_method(in_dim, out_dim)
            end    
            W = Variable(W, name="W")
            b = Variable(zeros(1, out_dim), name="b")
            layer = new(W, b, in_dim, out_dim, initial_method)
            return layer
        end
    end
end


parameters(linear::Linear) = (linear.W, linear.b)



function forward(layer::Linear, x)
    if layer.W isa Nothing
        layer.in_dim = size(x.values)[2]
        if layer.initial_method == "xavier"
            W = xavier(layer.in_dim, layer.out_dim)
        elseif layer.initial_method == "he"
            W = he(layer.in_dim, layer.out_dim)
        else layer.initial_method isa Function
            W = layer.initial_method(layer.in_dim, layer.out_dim)
        end   
        layer.W = Variable(W, name="W")
    end 
    return linear(x, layer.W, layer.b)
end

