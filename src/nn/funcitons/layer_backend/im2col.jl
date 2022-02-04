using PaddedViews

function im2col(X::AbstractArray{T, 4}, W_f::Int, H_f::Int, S::Int, P::Int) where T
    N, C, H, W = size(X)
    if P != 0
        # By using `PaddedView`, padding can be done without copying  
        X = PaddedView(zero(T), X, (1:N, 1:C, 1:H+2P, 1:H+2P), (1:N, 1:C, P+1:P+H, P+1:P+W))
    end
    H_o = (H + 2P - H_f) ÷ S + 1
    W_o = (W + 2P - W_f) ÷ S + 1
    col = zeros(T, (N, C, H_f, W_f, H_o, W_o))
    for y in 1:H_f
        y_max = y + S * H_o - 1
        for x in 1:W_f
            x_max = x + S * W_o - 1
            col[:, :, x, y, :, :] = X[:, :, x:S:x_max, y:S:y_max]
        end
    end
    return reshape(permutedims(col, (6, 2, 1, 5, 4, 3)), (W_f * H_f * C, W_o * H_o * N))
end
