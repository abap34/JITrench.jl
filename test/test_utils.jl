using Primes

function randfact(N; max_len=10)
    l = rand(1:max_len)
    base_arr = factor(Array, N)
    ind = rand(1:l, length(base_arr))
    return prod.(getindex.(Ref(base_arr),　(i -> ind .== i).(1:l)))
end


function generate_shape(n_elemtnt; max_dim=5)
    return Tuple(randfact(n_elemtnt, max_len=max_dim)), Tuple(randfact(n_elemtnt, max_len=max_dim))
end