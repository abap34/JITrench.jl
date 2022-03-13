using Printf

operators = [+, -, /, *]
N_TEST_COUNT = 1

@testset "OperatorTest" begin
    @testset "ForwardTest" begin
        for i in 1:N_TEST_COUNT
            for op in operators
                x1 = rand()
                x2 = rand()
                y = op(x1, x2)
                y_var = op(Variable(x1), Variable(x2))
                @test y ≃ y_var.values
            end
        end
    end
    @testset "backwardTest" begin
        for i in 1:N_TEST_COUNT
            for op in operators
                x1 = rand()
                x2 = rand()
                numerical_grad = numerical_diff(op, [x1, x2])
                backward_grad = backward_diff(op, [x1, x2])
                @test numerical_grad ≃ backward_grad
            end
        end
    end
end

funcitons = [
    sin,
    cos,
    tan,
    log,
    exp
]

@testset "MathFuncTest" begin
    @testset "ForwardTest" begin
        for i in 1:N_TEST_COUNT
            for f in funcitons
                x = rand()
                y = f(x)
                y_var = f(Variable(x))
                @test y ≃ y_var.values
            end
        end
    end
    @testset "backwardTest" begin
        for i in 1:N_TEST_COUNT
            for op in funcitons
                x = rand()
                numerical_grad = numerical_diff(op, x)
                backward_grad = backward_diff(op, x)
                @test numerical_grad ≃ backward_grad
            end
        end
    end  
end



@testset "NormalFuncTest" begin
    f1(x) = (2x + 3x) / sin(x) + 2
    f2(x) = ((1 + x) + x) / x * x
    f3(x) = (sin(x) + cos(x)) / 2
    f4(x) = x + sin(x) + log(x)
    f5(x) = 3x^2 + 10x + 6
    functions = (
        f1,
        f2,
        f3,
        f4,
        f5
    )
    @testset "ForwardTest" begin
        for i in 1:N_TEST_COUNT
            for f in funcitons
                x = rand()
                y = f(x)
                y_var = f(Variable(x))
                @test y ≃ y_var.values
            end
        end
    end
    @testset "backwardTest" begin
        for i in 1:N_TEST_COUNT
            for op in funcitons
                x = rand()
                numerical_grad = numerical_diff(op, x)
                backward_grad = backward_diff(op, x)
                @test numerical_grad ≃ backward_grad
            end
        end
    end  
end
