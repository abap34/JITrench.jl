struct BroadcastWrapper{F <: DiffableFunction} <: DiffableFunction 
    wrapped_func :: F
end

