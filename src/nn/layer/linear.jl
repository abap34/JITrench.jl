import ..JITrench: forward, backward, GradField

struct LinearFn <: DiffableFunction
    grad_field :: GradField
end

function JITrench.forward(::Type{LinearFn}, x, W, b) 
    return x * W .+ b
end

function JITrench.backward(f::LinearFn, gy)
    x, W, b = f.grad_field.inputs
    gx = gy * transpose(W)
    gW = transpose(x) * gy
    gb = AutoDiff.sum_to(gy, size(b))
    return gx, gW, gb
end

linearfn(x, W, b) = JITrench.call!(LinearFn, x, W, b)

struct Linear <: Layer 
    in_dim :: Union{Nothing, Int}
    out_dim :: Int
    inital_method :: String
    function Linear(;in_dim::Union{Nothing, Int}=nothing, out_dim::Int, initial_method::String="xavier")
        new(in_dim, out_dim, initial_method)
    end
end

function (linear::Linear)(initializer::Initializer)
    in_dim = initializer.current_shape[2]
    device = initializer.device
    if !(linear.in_dim isa Nothing)
        if in_dim != linear.in_dim
            throw(DimensionMismatch("Input dimension $in_dim does not match the expected dimension $(linear.in_dim)"))
        end
    end
    out_dim = linear.out_dim
    if linear.inital_method == "xavier"
        W = xavier(in_dim, out_dim)
    elseif linear.inital_method == "he"
        W = he(in_dim, out_dim)
    else linear.inital_method isa Function
        W = linear.nitial_method(in_dim, out_dim)
    end
    
    if device isa CPU
        W = Tensor(W, name="W")
        b = Tensor(zeros(1, out_dim), name="b")
    elseif device isa GPU
        W = CuTensor(W, name="W")
        b = CuTensor(zeros(1, out_dim), name="b")
    end

    register!(
        initializer, 
        Linear, 
        Dict(
            "W" => W,
            "b" => b
        )
    )

    initializer.current_shape[2] = out_dim
    return initializer    
end


function apply(::Linear, weight::Dict{String, <: AbstractTensor}, x)
    W = weight["W"]
    b = weight["b"]
    return linearfn(x, W, b)
end


