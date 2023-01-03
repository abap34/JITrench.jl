macro diff!(ex)
    f_name = ex.args[1]
    df_name = Symbol(String(f_name) * "′")
    quote
        function $(esc(df_name))(x)
            x_var = JITrench.AutoDiff.Scalar(x)
            y = $(esc(ex.args[1]))(x_var)
            backward!(y)
            return x_var.grad
        end
    end
end