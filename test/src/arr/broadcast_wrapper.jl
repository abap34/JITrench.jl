@testset "UserDefinedFunctionBroadcastTest" begin
    funcs = (
        (x -> x + 1), 
        (x -> 3x + 1), 
        (x -> 3x^2 + 5x + 1), 
        (x -> exp(x) - 10), 
        (x -> -x), 
        (x -> sin(x) + cos(sin(x)))
    ) 
    for f in funcs
        x = rand(10)
        y = f.(x)
        x_var = Variable(x)
        y_var = f.(x_var)
        @test y == y_var.values
        @test x_var.req_broadcast == false
    end
end
