using BenchmarkTools
using JITrench

SUITE = BenchmarkGroup()
x = AutoDiff.Tensor(rand(100))
SUITE["sum"] = @benchmarkable sum(x)