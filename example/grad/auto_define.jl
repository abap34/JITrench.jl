using  JITrench

# This macro automatically define the derivatives function.
macro diffable(ex)
    eval(ex)
    func_name = Meta.parse(String(ex.args[1].args[1]) * ("′"))
    @eval function $func_name(args)
        args_var = Variable(args)
        y = $(ex.args[1].args[1])(args_var)
        backward!(y)
        return args_var.grad.values
    end
end


@diffable f(x) = 3x^2 + 4x + 3
# f′(x) = 6x + 4

@show f′(1)
@show f′(3)



