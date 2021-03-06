using  JITrench

# This macro automatically define the derivatives function.
macro diff!(ex)
    f_name = ex.args[1]
    df_name = Symbol(String(f_name) * "′")
    quote
        function $(esc(df_name))(x)
            x_var = Variable(x)
            y = $(esc(ex.args[1]))(x_var)
            backward!(y)
            return x_var.grad.values
        end
    end
end

f(x) = 2x^2 + 4x + 3

@diff! f(x) 
# f′(x) = 4x + 4

f′(2)
# 12

f′(3)
# 16



