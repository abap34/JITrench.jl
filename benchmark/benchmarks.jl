using BenchmarkTools
using JITrench

SUITE = BenchmarkGroup()
x = AutoDiff.Tensor(rand(10000))
SUITE["sum"] = @benchmarkable sum(x)