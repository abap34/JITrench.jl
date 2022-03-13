# ==== check simple graph ====

x = Variable(10)
y = x + 2

# check x 
@test x.creator isa Nothing
@test x.generation == 0
@test x.name isa Nothing
@test x.grad isa Nothing


# check y
@test y.creator isa JITrench.Add
@test y.values == 12
@test y.generation == 1
@test y.name isa Nothing

# check y.creator
f = y.creator
@test length(f.grad_field.inputs) == 2
@test (x -> x.values).(f.grad_field.inputs) == [10, 2]
@test length(f.grad_field.outputs) == 1
@test f.grad_field.generation == 0

# check backward!

# retain_grad = false
backward!(y)
@test y.grad isa Nothing
@test x.grad.values == 1

# retain_grad = true
x = Variable(10)
y = x + 2
backward!(y, retain_grad=true)
@test y.grad.values == 1
@test x.grad.values == 1
# ============================


# ==== check more complex case ====

x = Variable(10)
y = Variable(2)
z = (x + y) / (x * y) + x
# to watch this compution as graph, see test/utils/core_test_graph.png

backward!(z) # retain_grad = false
@test z.grad isa Nothing
@test z.creator.grad_field.inputs[1].grad isa Nothing


x = Variable(10)
y = Variable(2)
z = (x + y) / (x * y) + x
backward!(z, retain_grad = true)

# check grad
@test !(z.grad.values isa Nothing)
@test !(z.creator.grad_field.inputs[1].grad isa Nothing)


# =================================
