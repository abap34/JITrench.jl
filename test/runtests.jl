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
        "broadcast.jl",
        "flatten.jl",
        "broadcast_wrapper.jl"
    ]
)


for (dir, files) in test_files
    @testset "$dir" begin
        for file in files
            @testset "$file" begin
                include("src/" * dir * "/" * file)
            end
        end
    end
end
