using JITrench: get_values, ones_like, cleargrad!, as_tuple


# get_values
x = Variable(10)
@test get_values(x) == 10
@test get_values(10) == 10

# ones_like
@test ones_like(0.5) ==  1.0
@test ones_like(0) == 1
@test ones_like([0, 0, 0]) == [1, 1, 1]
@test ones_like([0 1 2; 2 3 4]) == [1 1 1; 1 1 1]

# cleargrad!
cleargrad!(x)
@test x.grad isa Nothing
y = x + 1
backward!(y)
cleargrad!(x)
@test x.grad isa Nothing


# as_tuple

@test as_tuple(1) == (1, )
@test as_tuple((1, 1)) == (1, 1)