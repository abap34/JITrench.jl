struct NotImplemetedError <: Exception 
    msg :: String
end

Base.showerror(io::IO, e::NotImplemetedError) = print(io, e.msg)