# (func, n_args)
broadcastable_functions =  [
    (+, 2),
    (-, 2),  # subtraction (a - b)
    (-, 1),  # negative number (-b)
    (*, 2),
    (/, 2),
    (log, 1),
    (exp, 1),
    (tan, 1),
    (cos, 1),
    (sin, 1)
]

@testset "BroadCastForwardTest" begin
    for func in broadcastable_functions
        if func[2] == 2
            # test args is scalar and vector
            arg1 = randn()
            arg2 = randn(50)
            y = func[1].(arg1, arg2)
            y_jt = func[1].(Variable(arg1), Variable(arg2))
            @test y == y_jt.values

            # test args are vector
            args = randn(100)
            arg1 = args[1:50]
            arg2 = args[51:100]
            y = func[1].(arg1, arg2)
            y_jt = func[1].(Variable(arg1), Variable(arg2))
            @test y == y_jt.values
        end
    end
end

@testset "BroadCastBackwardTest" begin
    for func in broadcastable_functions
        if func
end