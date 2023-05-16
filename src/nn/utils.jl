using JLD2

JLD2.writeas(::Type{Tensor{T}}) where T = T
JLD2.wconvert(::Type{<: AbstractArray}, x::AbstractTensor) = x.values
JLD2.rconvert(::Type{Tensor{T}}, x::T) where T = Tensor(x)

function save_weight(parameter::Parameter, filename::AbstractString)
    jldsave(filename * ".jtw"; parameter)
    return
end

function load_weight(filename::AbstractString)
    return jldopen(filename * ".jtw")["parameter"]
end