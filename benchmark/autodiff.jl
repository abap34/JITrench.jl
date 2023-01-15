
SUITE["AutoDiff"] = BenchmarkGroup(["AutoDiff", "core"])
SUITE["AutoDiff"]["scalar_create"] = BenchmarkGroup(["scalar"])
SUITE["AutoDiff"]["Tensor_create"] = BenchmarkGroup(["tensor"])

SUITE["AutoDiff"]["scalar_create"]["Int"] = @benchmarkable AutoDiff.Scalar(1)
SUITE["AutoDiff"]["scalar_create"]["Float"] = @benchmarkable AutoDiff.Scalar(0.1)

A = rand(Int, 10, 10)
SUITE["AutoDiff"]["Tensor_create"]["Int"] = @benchmarkable AutoDiff.Tensor($A)
B = rand(Float64, 10, 10)
SUITE["AutoDiff"]["Tensor_create"]["Float"] = @benchmarkable AutoDiff.Tensor($B)

unary_scalar_functions = Dict(
    "neg" => -,
    "sin" => sin,
    "cos" => cos,
    "tan" => tan,
    "log" => log,
    "exp" => exp,
    "square" => (x -> x^2),
    "sqrt" => sqrt
)

SUITE["AutoDiff"]["simple_call"] = BenchmarkGroup()
SUITE["AutoDiff"]["complex_call"] = BenchmarkGroup()
SUITE["AutoDiff"]["simple_call"]["scalar"] = BenchmarkGroup(["scalar"])
SUITE["AutoDiff"]["complex_call"]["scalar"] = BenchmarkGroup(["scalar"])

r = AutoDiff.Scalar(rand())
for (operator_name, operator) in unary_scalar_functions
    SUITE["AutoDiff"]["simple_call"]["scalar"][operator_name] = @benchmarkable $operator($r)
end

binary_scalar_functions = Dict(
    "add" => +,
    "sub" => -,
    "mul" => *,
    "div" => /,
)

r1 = AutoDiff.Scalar(rand())
r2 = AutoDiff.Scalar(rand())
for (operator_name, operator) in binary_scalar_functions
    SUITE["AutoDiff"]["simple_call"]["scalar"][operator_name] = @benchmarkable $operator($r1, $r2)
end

ackley(x, y) = -20 * exp(-0.2 * sqrt((x^2 + y^2) / 2)) - exp((cos(2 * π * x) + cos(2 * π * y)) / 2) + ℯ + 20
goldstain(x, y) = (1 + (x + y + 1)^2 * (19 - 14x + 3x^2 - 14y + 6x * y + 3y^2)) * (30 + (2x - 3y)^2 * (18 - 32x + 12x^2 + 48y - 36x * y + 27 * y^2))


SUITE["AutoDiff"]["complex_call"]["scalar"]["ackley"] = @benchmarkable ackley($r1, $r2)
SUITE["AutoDiff"]["complex_call"]["scalar"]["goldstain"] = @benchmarkable goldstain($r1, $r2)