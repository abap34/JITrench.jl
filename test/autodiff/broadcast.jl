unary_scalar_functions = Dict(
    "neg" => -, 
    "sin" => sin,
    "cos" => cos, 
    "tan" => tan,
    "log" => log,
    "exp" => exp, 
    "square" => (x -> x^2),
    "sqrt" => sqrt,
    "inv" => inv
)

@testset "UnaryForwardTest" begin
    for (operator_name, operator) in unary_scalar_functions
        @testset "$operator_name" begin
            for i in 1:N_test
                x = rand(rand(1:30))
                if !(check_close(operator.(x), operator.(Tensor(x)).values, atol=1e-10))
                    @info "x: $x"
                end
            end
        end
    end
end

@testset "UnaryFunctionTest" begin
    for (operator_name, operator) in unary_scalar_functions
        @testset "$operator_name" begin
            @testset "ForwardTest" begin
                for i in 1:N_test
                    x = rand(rand(1:30))
                    if !(check_close(operator.(x), operator.(Tensor(x)).values, atol=1e-10))
                        @info "x: $x"
                    end
                end
            end
            @testset "BackWardTest" begin
                for i in 1:N_test
                    x = rand(rand(1:30))
                    func = (x -> sum(operator.(x)))
                    gx_numerical = numerical_grad(func, x)
                    gx_backprop = backprop_grad(func, x)
                    if !(check_close(gx_numerical, gx_backprop, atol=1e-5))
                        @info "x: $x"
                    end
                end
            end
        end
    end
end
