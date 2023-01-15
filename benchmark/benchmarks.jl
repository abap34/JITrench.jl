using BenchmarkTools
using JITrench

using Random
Random.seed!(34)


SUITE = BenchmarkGroup()

include("autodiff.jl")
