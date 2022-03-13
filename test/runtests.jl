using Test
using JITrench


include("./test_utils.jl")


test_files = Dict(
    "core" => [
        "functions_utils.jl",
        "scalar_gradcheck.jl",
        "propagation.jl"
    ],
    "arr" => [
        "broadcast_to.jl",
        "broadcast.jl"
    ]
)


for (dir, files) in test_files
    @testset "$dir" begin
        for file in files
            @testset "$file" begin
                # println("test: ", "test/src/" * dir * "/" * file)
                include("src/" * dir * "/" * file)
            end
        end
    end
end



# using Random
# using StatsBase
# using Printf
# using JITrench

# include("test_utils.jl")

# operators = [+, -, /, *]

# function test_all(
#     N_TEST_COUNT = 100
# )
# @testset "OperatorTest" begin
#     @testset "ForwardTest" begin
#         for i in 1:N_TEST_COUNT
#             for op in operators
#                 x1 = rand()
#                 x2 = rand()
#                 @test isAbout(op(Variable(x1), Variable(x2)).values, op(x1, x2), )
#             end
#         end
#     end
#     @testset "backwardTest" begin
#         for i in 1:N_TEST_COUNT
#             for op in operators[1:end-1]
#                 x1 = rand()
#                 x2 = rand()
#                 @test isAbout(backward_diff(op, [x1, x2]), numerical_diff(op, [x1, x2]))
#             end
#         end
#     end
# end

# funcitons = [
#     sin,
#     cos,
#     tan,
#     log,
#     exp
# ]

# @testset "MathFuncTest" begin
#     @testset "ForwardTest" begin
#         for i in 1:N_TEST_COUNT
#             for op in funcitons
#                 x = rand()
#                 @test isAbout(op(x), op(Variable(x).values))
#             end
#         end
#     end
#     @testset "backwardTest" begin
#         for i in 1:N_TEST_COUNT
#             for op in funcitons
#                 x = rand()
#                 @test isAbout(backward_diff(op, [x]), numerical_diff(op, [x]))
#             end
#         end
#     end  
# end

# activations = Dict(
#     JITrench._sigmoid => JITrench.sigmoid 
# )

# @testset "ActivationTest" begin
#     @testset "ForwardTest" begin
#         for i in 1:N_TEST_COUNT
#             for (func, jt_func) in activations
#                 x = rand()
#                 isAbout(func(x), jt_func(Variable(x)).values)
#             end
#         end
#     end
#     @testset "backwardTest" begin
#         for i in 1:N_TEST_COUNT
#             for (func, jt_func) in activations
#                 x = rand()
#                 @test isAbout(backward_diff(jt_func, [x]), numerical_diff(func, [x]))
#             end
#         end
#     end  
# end


# @testset "ArrOperateTest" begin
#     max_size = 120
#     @testset "forward" begin
#         @testset "reshape" begin
#             for _ in 1:N_TEST_COUNT
#                 in_shape, out_shape = generate_shape(rand(1:max_size))
#                 x = rand(in_shape...)
#                 y_arr = reshape(x, out_shape)
#                 y_var = reshape(Variable(x), out_shape)
#                 @test y_arr == y_var.values
#             end
#         end
#         @testset "transpose" begin
#             for _ in 1:N_TEST_COUNT
#                 in_shape, out_shape = generate_shape(rand(1:max_size), max_dim=2)
#                 x = rand(in_shape...)
#                 y_arr = transpose(x)
#                 y_var = transpose(Variable(x))
#                 @test y_arr == y_var.values
#             end
#         end
#     end 
#     @testset "backward" begin
#         @testset "reshape" begin
#             for _ in 1:N_TEST_COUNT
#                 in_shape, out_shape = generate_shape(rand(1:max_size))
#                 x = Variable(rand(in_shape...))
#                 y = reshape(x, out_shape)
#                 backward!(y, retain_grad=true)
#                 @test y.grad.values == ones(size(y.values))
#                 @test x.grad.values == ones(size(x.values))
#             end
#         end
#         @testset "transpose" begin
#             for _ in 1:N_TEST_COUNT
#                 in_shape, out_shape = generate_shape(rand(1:max_size), min_dim=2, max_dim=2)
#                 x = Variable(rand(in_shape...))
#                 y = JITrench.transpose(x)
#                 backward!(y, retain_grad=true)
#                 @test y.grad.values == ones(size(y.values))
#                 @test x.grad.values == ones(size(x.values))
#             end
#         end
#     end
# end


# end
