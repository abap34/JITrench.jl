@testset "FlattenForwardTest" begin
    n_testcase = 10
    for i in 1:n_testcase
        n_element = rand(1:120)
        shape = randshape(n_element)
        arr = rand(1:20, shape)
        @test flatten(Variable(arr)).values == vcat(arr...)
    end
end