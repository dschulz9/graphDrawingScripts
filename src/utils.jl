@inline function initialize_distance_matrix(adjacencies::Array{Float64, 3})
    D = copy(adjacencies)
    for I in CartesianIndices(D)
        if I[1] != I[2] && D[I] <= 0.0
            D[I] = Inf
        end
    end

    return D
end

function pairwise_distances_per_slice(adjacencies::Array{Float64,3})
    n, _, K = size(adjacencies)
    D = initialize_distance_matrix(adjacencies)

    for k in 1:K
        for mid in 1:n
            D[:, :, k] = min.(D[:, mid, k] .+ D[mid, :, k]', D[:, :, k])
        end
    end

    return D
end

function gram_schmidt(u::Vector{Float64})
    d = length(u)

    T = zeros(Float64, d, d)
    T[1, :] .= u
    num_rows = 1
    for i in 1:d
        w = [1.0 * (i == k) for k in 1:d]
        for j in 1:num_rows
            w -= T[j, i] * view(T, j, :)
        end

        w_norm = norm(w)
        if w_norm > 1e-9
            num_rows += 1
            w /= norm(w)
            T[num_rows, :] .= w
        end
    end

    return T
end

