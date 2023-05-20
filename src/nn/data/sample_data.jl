using MLDatasets

function mnist(train_sample=3000, val_sample=1000)
    train_x, train_y = MNIST(split=:train)[:]
    val_x, val_y = MNIST(split=:test)[:]

    train_x = transpose(reshape(train_x, (28 * 28, :)))
    val_x = transpose(reshape(val_x, (28 * 28, :)))

    train_x = train_x[1:train_sample, :]
    train_y = train_y[1:train_sample]
    val_x = val_x[1:val_sample, :]
    val_y = val_y[1:val_sample]

    train_x = Tensor(Float64.(train_x))
    train_y = Tensor(Float64.(train_y))

    val_x = Tensor(Float64.(val_x))
    val_y = Tensor(Float64.(val_y))

    return train_x, train_y, val_x, val_y
end