binary_scalar_functions = Dict(
    "add" => +, 
    "sub" => -,
    "mul" => *, 
    "div" => /,
)



@testset "BinaryFunctionTest" begin
    for (operator_name, operator) in binary_scalar_functions
        @testset "$operator_name" begin
            @testset "ForwardTest" begin
                for i in 1:N_test
                    x1 = randn()
                    x2 = randn()
                    if !(check_close(operator(x1, x2), operator(AutoDiff.Scalar(x1), AutoDiff.Scalar(x2)).values))
                        @info "x1: $x1 x2:$x2"
                    end
                end
            end
            @testset "BackWardTest" begin
                for i in 1:N_test
                    x1 = randn()
                    x2 = randn()
                    gx_numerical = numerical_grad(operator, x1, x2)
                    gx_backprop = backprop_grad(operator, x1, x2)
                    if !(check_close(gx_numerical, gx_backprop))
                        @info "x1: $x1 x2:$x2"
                    end
                end
            end
        end
    end
end
