function xavier(in_dim, out_dim)
    return randn(in_dim, out_dim) ./ sqrt(in_dim)
end


function he(in_dim, out_dim)
    return randn(in_dim, out_dim) ./ sqrt(2 / in_dim)
end