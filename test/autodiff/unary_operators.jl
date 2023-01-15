unary_scalar_functions = Dict(
    "neg" => -, 
    "sin" => sin,
    "cos" => cos, 
    "tan" => tan,
    "log" => log,
    "exp" => exp, 
    "square" => (x -> x^2),
    "sqrt" => sqrt,
)


@testset "UnaryFunctionTest" begin
    for (operator_name, operator) in unary_scalar_functions
        @testset "$operator_name" begin
            @testset "ForwardTest" begin
                for i in 1:N_test
                    x = rand()
                    if !(check_close(operator(x), operator(AutoDiff.Scalar(x)).values))
                        @info "x: $x"
                    end
                end
            end
            @testset "BackWardTest" begin
                for i in 1:N_test
                    x = rand()
                    gx_numerical = numerical_grad(operator, x)
                    gx_backprop = backprop_grad(operator, x)
                    if !(check_close(gx_numerical, gx_backprop))
                        @info "x: $x"
                    end
                end
            end
        end
    end
end
