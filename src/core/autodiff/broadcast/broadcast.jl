struct BroadcastWrapper{F <: DiffableFunction} <: DiffableFunction 
    wrapped_func :: F
end

_get_gf(wrapper::BroadcastWrapper) = wrapper.wrapped_func.grad_field