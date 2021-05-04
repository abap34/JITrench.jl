using Test
using Random
using StatsBase
using Printf
using JITrench



function generate_expr(N)
    randop() = sample(["+", "-", "/", "*", "^"])
    randnum(scale=100) = @sprintf("%.2f", scale * (rand() - 0.5))
    randarg() = rand() > 0.5 ? "x" : randnum()
    S = "$(randarg()) $(randop()) $(randarg())"
    N = 10
    for i in 1:N
        x1, x2 = randnum(), randnum()
        op = randop()
        if op == "^"
            x2 = rand(1:10)
        elseif op == "/"
            x2 = randnum() 
        end
        S = "($S) $(randop()) $(randarg())"
    end
    return S
end


function numerical_diff(f::Function, x::Real; e=1e-7)
    return (f(x + e) - f(x - e)) / 2e
end


function numerical_diff(f::Function, xs::AbstractArray; e=1e-7)
    grads = zeros(length(xs)) 
    for idx in 1:length(xs)
        tmp_val = xs[idx]
        xs[idx] = tmp_val + e
        fxh1 = f(xs...)
        xs[idx] = tmp_val - e
        fxh2 = f(xs...)
        grads[idx] = (fxh1 - fxh2) / 2e
        xs[idx] = tmp_val
    end
    return grads
end


function backward_diff(f::Function, xs::AbstractArray)
    inputs = Variable.(xs)
    outs = f(inputs...)
    backward!(outs)
    return (input -> input.grad.values).(inputs)
end

function isAbout(x, y; e=1e-4)
    return ((x - y) < 1e-15) || (-e < ((x - y) / y) < e)
end

function isAbout(X::AbstractArray, Y::AbstractArray; e=1e-4)
    for (x, y) in zip(X, Y)
        if !(((x - y) < 1e-15) || (-e < ((x - y) / y) < e))
            return false
        end
    end
    return true
end

operators = [+, -, /, *, ^]

N_TEST_COUNT = 1

@testset "OperatorTest" begin
    @testset "ForwardTest" begin
        for i in 1:N_TEST_COUNT
            for op in operators
                x1 = rand()
                x2 = rand()
                @test isAbout(op(Variable(x1), Variable(x2)).values, op(x1, x2), )
            end
        end
    end
    @testset "backwardTest" begin
        # test without ^
        for i in 1:N_TEST_COUNT
            for op in operators[1:end-1]
                x1 = rand()
                x2 = rand()
                @test isAbout(backward_diff(op, [x1, x2]), numerical_diff(op, [x1, x2]))
            end
            # test ^
            x1 = rand()
            x2 = rand(1:10)
            f(x) = x ^ x2
            num_grad = numerical_diff(f, x1)
            x1 = Variable(x1)
            x2 = Variable(x2)
            y = x1 ^ x2
            backward!(y)
            @test isAbout(x1.grad.values, num_grad)
        end
    end
end

funcitons = [
    sin,
    cos,
    tan
]

@testset "MathFuncTest" begin
    @testset "ForwardTest" begin
        for i in 1:N_TEST_COUNT
            for op in funcitons
                x = rand()
                @test isAbout(op(x), op(Variable(x).values))
            end
        end
    end
    @testset "backwardTest" begin
        for i in 1:N_TEST_COUNT
            for op in funcitons
                x = rand()
                @test isAbout(backward_diff(op, [x]), numerical_diff(op, [x]))
            end
        end
    end  
end