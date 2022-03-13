using Test
using JITrench

x = Variable(1)

y = JITrench.broadcast_to(x, (2, 2))

@test y.values == [1 1; 1 1]
@test y.creator isa JITrench.BroadcastTo
@test y.creator.shape == (2, 2)
@test y.creator.x_shape == () 

backward!(y, retain_grad=true)

@test size(y.grad) == size(y)
@test size(x.grad) == size(x)
@test y.grad.values == [1 1; 1 1]
@test x.grad.values == 4


x = Variable([1, 2])

y = JITrench.broadcast_to(x, (2, 2))

@test y.values == [1 1; 2 2]
@test y.creator isa JITrench.BroadcastTo
@test y.creator.shape == (2, 2)
@test y.creator.x_shape == (2, ) 

backward!(y, retain_grad=true)

@test size(y.grad) == size(y)
@test size(x.grad) == size(x)
