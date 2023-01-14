function numerical_grad(f::Function, x::Real; dx=1e-7) 
    return (f(x+dx) - f(x-dx)) / 2dx
end

function numerical_grad(f::Function, x1::Real, x2::Real; dx=1e-7) 
    gx1 = (f(x1+dx, x2) - f(x1-dx, x2)) / 2dx
    gx2 = (f(x1, x2+dx) - f(x1, x2-dx)) / 2dx
    return gx1, gx2
end

function backprop_grad(f::Function, x::Real)
    x_var = AutoDiff.Scalar(x)
    y = f(x_var)
    backward!(y)
    return x_var.grad
end

function backprop_grad(f::Function, x1::Real, x2::Real)
    x1_var = AutoDiff.Scalar(x1)
    x2_var = AutoDiff.Scalar(x2)
    y = f(x1_var, x2_var)
    backward!(y)
    return x1_var.grad, x2_var.grad
end


# note that arguments must be vector of Float
function numerical_grad(f::Function, X::Vector{Float64}, dx=1e-7) 
    n = length(X)
    grads = zeros(n)
    for i in eachindex(X)
        X[i]  = X[i] + dx
        y_f = f(X...)
        X[i] = X[i] - dx
        y_b = f(X...)
        X[i] = X[i] + dx  
        grads[i] = (y_f - y_b) / 2dx
    end
    return grads
end




function check_close(x1, x2, atol=1e-6)
    # scaling
    x1_scaled = x1 ./ max(x1, x2)
    x2_scaled = x2 ./ max(x1, x2)
    return onfail(@test all(isapprox.(x1_scaled, x2_scaled, atol=1e-6))) do 
        @info "x1:$x1 x2:$x2. error_rate = $((x1_scaled .- x2_scaled))"
        return false
    end
end


function check_close(x1::Real, x2::Real, atol=1e-6)
    # scaling
    x1_scaled = x1 / max(x1, x2)
    x2_scaled = x2 / max(x1, x2)
    return onfail(@test isapprox(x1_scaled, x2_scaled, atol=1e-6)) do 
        @info "x1:$x1 x2:$x2. error_rate = $((x1_scaled - x2_scaled))"
        return false
    end
end
    
    

onfail(body, _::Test.Pass) = true

onfail(body, _::Union{Test.Fail, Test.Error}) = body()
