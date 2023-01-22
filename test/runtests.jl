using JITrench
using Test

include("./test_utils.jl")


test_files = Dict(
    "autodiff" => [
        "unary_operators.jl",
        "binary_operators.jl",
        "broadcast.jl"
    ]
)



N_test = 100


for (dir, files) in test_files
    @testset "$dir" begin
        for file in files
            @testset "$file" begin
                path = dir * "/" * file
                @info "Testing $path"
                include(path)
            end
        end
    end
end


