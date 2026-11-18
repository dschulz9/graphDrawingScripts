using Enzyme, Optimisers, Random, LinearAlgebra

include("utils.jl")

# @inline function project_onto_orthogonal_complement(x, u)
#     return x - dot(x, u) * u
# end

@inline function projected_distance(X::Array{Float64,2}, U::Array{Float64,2}, i::Int64, j::Int64, k::Int64)
    inner_prod = 0.0
    d = size(X, 1)
    # @inbounds for l in 1:d
    for l in 1:d
        inner_prod += (X[l, i] - X[l, j]) * U[l, k]
    end

    s = 0.0

    # @inbounds for l in 1:d
    for l in 1:d
        projected_diff = X[l, i] - X[l, j] - inner_prod * U[l, k]
        s += projected_diff * projected_diff
    end

    return s
end

loss(
    D::Array{Float64,3},
    X::Array{Float64,2},
    U::Array{Float64,2},
    C::Float64,
    indices::SubArray{CartesianIndex{3}},
    p::Float64
) = loss_squared_stabilized(D, X, U, C, indices, p)

function loss_norm(D::Array{Float64,3}, X::Array{Float64,2}, U::Array{Float64,2}, C::Float64, indices::SubArray{CartesianIndex{3}}, p::Float64)
    n, _, K = size(D)

    loss = 0.0
    for I in indices[k]
        i = I[1]
        j = I[2]
        k = I[3]

        loss += (D[i, j, k] - norm(project_onto_orthogonal_complement(view(X, :, i) - view(X, :, j), view(U, :, k))))^2
    end

    return loss/(C * p)
end

function loss_squared(D::Array{Float64,3}, X::Array{Float64,2}, U::Array{Float64,2}, C::Float64, indices::SubArray{CartesianIndex{3}}, p::Float64)
    loss = 0.0
    for I in indices
        i = I[1]
        j = I[2]
        k = I[3]

        # projection = project_onto_orthogonal_complement(view(X, :, i) - view(X, :, j), view(U, :, k))
        # loss += (D[i, j, k]^2 - dot(projection, projection))^2
        loss += (D[i, j, k]^2 - projected_distance(X, U, i, j, k))^2
    end

    return loss/(C * p)
end

function loss_squared_stabilized(D::Array{Float64,3}, X::Array{Float64,2}, U::Array{Float64,2}, C::Float64, indices::SubArray{CartesianIndex{3}}, p::Float64)
    n, _, K = size(D)

    loss = 0.0
    for I in indices
        i = I[1]
        j = I[2]
        k = I[3]

        # projection = project_onto_orthogonal_complement(view(X, :, i) - view(X, :, j), view(U, :, k))
        # loss += (D[i, j, k]^2 - dot(projection, projection))^2
        loss += (D[i, j, k]^2 - projected_distance(X, U, i, j, k))^2
    end

    for I in indices
        if I[3] == K
            break
        end

        i = I[1]
        j = I[2]
        k = I[3]

        # projected_x_curr = project_onto_orthogonal_complement(view(X, :, i), view(U, :, k))
        # projected_x_next = project_onto_orthogonal_complement(view(X, :, i), view(U, :, k + 1))
        # projected_diff = projected_x_curr - projected_x_next
        # loss += dot(projected_diff, projected_diff)

        inner_prod_k_i = 0.0
        inner_prod_k_next_i = 0.0
        inner_prod_k_j = 0.0
        inner_prod_k_next_j = 0.0

        # @inbounds for l in axes(U, 1)
        for l in axes(U, 1)
            inner_prod_k_i += U[l, k] * X[l, i]
            inner_prod_k_j += U[l, k] * X[l, j]
            inner_prod_k_next_i += U[l, k+1] * X[l, i]
            inner_prod_k_next_j += U[l, k+1] * X[l, j]
        end

        # @inbounds for l in axes(U, 1)
        for l in axes(U, 1)
            projected_diff_i = inner_prod_k_next_i * U[l, k+1] - inner_prod_k_i * U[l, k]
            projected_diff_j = inner_prod_k_next_j * U[l, k+1] - inner_prod_k_j * U[l, k]
            loss += projected_diff_i * projected_diff_i
            loss += projected_diff_j * projected_diff_j
        end
    end

    return loss/(C * p)
end

function loss(D::Array{Float64,3}, X::Array{Float64,2}, U::Array{Float64,2}, C::Float64, relevant_indices::Vector{CartesianIndex{3}})
    n, _, K = size(D)

    loss = 0.0
    for I in relevant_indices
        i = I[1]
        j = I[2]
        k = I[3]

        # loss += (D[i, j, k] - norm(project_onto_orthogonal_complement(view(X, :, i) - view(X, :, j), view(U, :, k))))^2
        loss += (D[i, j, k]^2 - projected_distance(X, U, i, j, k))^2
    end

    return loss/C
end


function optimize_mu_(D::Array{Float64,3}, X::Array{Float64,2}, U::Array{Float64,2}, rng::AbstractRNG, p::Float64, max_steps::Int64, relevant_indices::Vector{CartesianIndex{3}})
    C = sum(D[findall(isfinite.(D))] .^ 2)
    n, _, K = size(D)
    losses = zeros(Float64, max_steps)

    X_old = zeros(Float64, size(X))
    X_curr = X

    U_old = zeros(Float64, size(X))
    U_curr = U

    mu_X = ones(Float64, n)
    mu_U = ones(Float64, K)


    dX_old = zeros(Float64, size(X))
    dU_old = zeros(Float64, size(U))

    dX_curr = zeros(Float64, size(X))
    dU_curr = zeros(Float64, size(U))

    for t in 1:max_steps
        fill!(dX_old, 0.0)
        fill!(dU_old, 0.0)
        indices = sample_indices_(rng, p, relevant_indices)
        autodiff(Enzyme.Reverse, loss, Enzyme.Active, Const(D), Duplicated(X_curr, dX_old), Duplicated(U_curr, dU_old), Const(C), Const(indices), Const(p))

        X_old = X_curr
        U_old = U_curr

        X_curr = X_curr - mu_X' .* dX_old
        U_curr = U_curr - mu_U' .* dU_old

        for k in 1:K
            U_curr[:, k] /= norm(view(U_curr, :, k))
        end

        losses[t] = loss(D, X_curr, U_curr, C, relevant_indices)
        norm_gradient = hypot(norm(dU_old), norm(dX_old))
        if losses[t] < 1e-9 || norm_gradient < 1e-9
            # @info "Loss: " losses[t]
            # @info "Norm gradient " norm_gradient
            return X_curr, U_curr, losses[1:t]
        end


        fill!(dX_curr, 0.0)
        fill!(dU_curr, 0.0)
        autodiff(Enzyme.Reverse, loss, Enzyme.Active, Const(D), Duplicated(X_curr, dX_curr), Duplicated(U_curr, dU_curr), Const(C), Const(indices), Const(p))

        mu_X = step_size_(X_curr, X_old, dX_curr, dX_old)
        mu_U = step_size_(U_curr, U_old, dU_curr, dU_old)
    end

    return X_curr, U_curr, losses
end

function optimize_(D::Array{Float64,3}, X::Array{Float64,2}, U::Array{Float64,2}, rng::AbstractRNG, p::Float64, max_steps::Int64, relevant_indices::Vector{CartesianIndex{3}})
    C = sum(D[findall(isfinite.(D))] .^ 2)
    n, _, K = size(D)
    losses = zeros(Float64, max_steps)

    rule = Optimisers.Adam()
    model = (X=X, U=U)
    grads = (X=zeros(Float64, size(X)), U=zeros(Float64, size(U)))

    state = Optimisers.setup(rule, model)
    sgd_loss = zeros(Float64, max_steps)

    last_time = time()
    indices = Vector{CartesianIndex{3}}(undef, length(relevant_indices))

    for t in 1:max_steps
        fill!(grads.X, 0.0)
        fill!(grads.U, 0.0)

        num_sampled = sample_indices_(rng, p, relevant_indices, indices)
        (_, curr_loss) = autodiff(ReverseWithPrimal, loss, Enzyme.Active, Const(D), Duplicated(model.X, grads.X), Duplicated(model.U, grads.U), Const(C), Const(view(indices, 1:num_sampled)), Const(p))
        sgd_loss[t] = curr_loss

        norm_gradient = hypot(norm(grads.X), norm(grads.U))

        state, model = Optimisers.update!(state, model, grads)

        for k in 1:K
            model.U[:, k] /= norm(view(model.U, :, k))
        end

        losses[t] = loss(D, model.X, model.U, C, relevant_indices)
        if losses[t] < 1e-9 || norm_gradient < 1e-9
            return model.X, model.U, losses[1:t], sgd_loss[1:t]
        end

        if mod(t, 100) == 0
            @info "$t iterations finished in $(time() - last_time) seconds"
            last_time = time()
        end
    end

    return model.X, model.U, losses, sgd_loss
end

@inline function step_size_(Y_curr::Array{Float64,2}, Y_old::Array{Float64,2}, grad_Y_curr::Array{Float64,2}, grad_Y_old::Array{Float64,2})
    Y_diff = Y_curr - Y_old
    grad_diff = grad_Y_curr - grad_Y_old
    numerator = [dot(view(Y_diff, :, j), view(grad_diff, :, j)) for j in axes(Y_diff, 2)]
    n = size(grad_diff)[2]
    denominator = ones(Float64, n)
    for j in 1:n
        mu = dot(view(grad_diff, :, j), view(grad_diff, :, j))
        if mu > 1e-9
            denominator[j] = mu
        end
    end
    return numerator ./ denominator
end

function grad_inititialization_(D::Array{Float64,3}, X::Array{Float64,2}, C::Float64, indices::SubArray{CartesianIndex{3}}, p::Float64)
    DX = zeros(size(X))
    for I in indices
        i = I[1]
        j = I[2]
        k = I[3]
        # delta_x = view(X, :, i) - view(X, :, j)
        # dist = norm(delta_x)
        # Dl = (dist - D[i, j, k])/dist * delta_x
        # DX[:, i] += Dl
        # DX[:, j] -= Dl

        dist = 0.0
        # @inbounds for l in axes(X, 1)
        for l in axes(X, 1)
            dist += (X[l, i] - X[l, j])^2
        end
        dist = sqrt(dist)

        # @inbounds for l in axes(X, 1)
        for l in axes(X, 1)
            Dl = (dist - D[i, j, k])/dist * (X[l, i] - X[l, j])
            DX[l, i] += Dl
            DX[l, j] -= Dl
        end
    end

    return 2/(C * p) * DX
end

@inline function sample_indices_(rng::AbstractRNG, p::Float64, relevant_indices::Vector{CartesianIndex{3}}, indices::Vector{CartesianIndex{3}})
    current_index = 0
    for I in relevant_indices
        if rand(rng) <= p
            current_index += 1
            indices[current_index] = I
        end
    end
    return current_index
end

@inline function sample_indices_(rng::AbstractRNG, p::Float64, n::Int64, indices::Vector{Int64})
    current_index = 0
    for I in 1:n
        if rand(rng) <= p
            current_index += 1
            indices[current_index] = I
        end
    end
    return current_index
end


function initialize_positions_(D::Array{Float64,3}, p::Float64, rng::AbstractRNG, max_steps::Int64, relevant_indices::Vector{CartesianIndex{3}})
    n, _, K = size(D)
    D_bar = zeros(Float64, (n, n))
    for I in relevant_indices
        i = I[1]
        j = I[2]
        D_bar[i, j] += D[I]
    end

    D_bar = 3/(2 * K) * D
    C = sum(D_bar .^ 2)

    X_old = zeros(Float64, 3, n)
    X_curr = rand(rng, 3, n) .- 0.5
    mu = ones(Float64, n)
    indices = Vector{CartesianIndex{3}}(undef, length(relevant_indices))

    for _ in 1:(max_steps/4)
        num_sampled = sample_indices_(rng, p, relevant_indices, indices)
        grad_X_old = grad_inititialization_(D, X_curr, C, view(indices, 1:num_sampled), p)

        X_old = X_curr
        X_curr -= mu' .* grad_X_old
        grad_X_curr = grad_inititialization_(D, X_curr, C, view(indices, 1:num_sampled), p)
        mu = step_size_(X_curr, X_old, grad_X_curr, grad_X_old)
    end

    @info "Positions initialized"
    return X_curr
end

function calculate_positions(adjacencies::Array{Float64,3}, max_steps::Int64, p::Float64, seed::Int64=123)
    D = pairwise_distances_per_slice(adjacencies)

    mask = isfinite.(D) .& (D .> 0)
    for k in axes(mask, 3)
        mask[:, :, k] = triu(mask[:, :, k])
    end
    relevant_indices = findall(mask)

    rng = Xoshiro(seed)
    m, n, K = size(D)
    @assert m == n
    @assert 0 < p <= 1
    X = initialize_positions_(D, p, rng, max_steps, relevant_indices)

    # Not uniform on the sphere
    U = zeros(Float64, 3, K)
    for i in 1:K
        U[:, i] = [cos((i * pi)/(K + 1)), sin((i * pi)/(K + 1)), 0]
    end

    X, U, losses, sgd_loss = optimize_(D, X, U, rng, p, max_steps, relevant_indices)
    views = Vector{Vector{Tuple{Float64,Float64}}}()

    for k in 1:size(U)[2]
        T = gram_schmidt(U[:, k])

        projected_X = T[2:end, :] * X
        projected_tuples = [Tuple(projected_X[:, j]) for j in axes(projected_X, 2)]
        push!(views, projected_tuples)
    end

    return [Tuple(X[:, j]) for j in axes(X, 2)], views, losses, sgd_loss, X, U
end

@inline function component_loss(component_positions::Array{Float64,2}, radii::Array{Float64,1}, indices::SubArray{Int64}, p::Float64)
    inner_taper = 2
    outer_taper = 1e-5

    loss = 0.0

    for i in indices
        for j in indices[i+1:end]
            dist = 0.0
            for k in axes(component_positions, 1)
                dist += (component_positions[k, i] - component_positions[k, j])^2
            end

            if dist <= radii[i]
                loss += inner_taper * (dist - radii[i] - radii[j])^2
            else
                loss += outer_taper * (dist - radii[i])^2
            end
        end
    end

    return loss/p
end

"""
We assume the graph has multiple connected components which will be arranged separately and then put into the same graph again.
"""
function place_unconnected_graph(adjacencies::Array{Float64,3}, max_steps::Int64, p::Float64, seed::Int64=123)
    G = SimpleGraph(adjacencies[:, :, end])
    components = connected_components(G)
    positions = Vector{Vector{Tuple{Float64,Float64,Float64}}}()
    # views = Vector{Vector{Vector{Tuple{Float64,Float64}}}}()

    for comp in components
        if length(comp) == 1
            push!(positions, [(0.0, 0.0, 0.0)])
            continue
        end
        pos, _, _, _, _, _ = calculate_positions(adjacencies[comp, comp, :], max_steps, p, seed)
        push!(positions, pos)
        # push!(views, v)
    end

    @info "Components placed"

    n = length(components)
    radii = zeros(Float64, n)
    for (i, comp) in enumerate(positions)
        max_dist = 0.0
        for point in comp
            max_dist = max(max_dist, dot(point, point) + 2)
        end

        radii[i] = 2 * max_dist
    end

    rng = Xoshiro(seed)
    component_positions = -10 .+ 20 * randn(rng, 3, n)
    indices = Vector{Int64}(undef, n)

    rule = Optimisers.Adam()
    model = component_positions
    grads = zeros(Float64, size(component_positions))

    state = Optimisers.setup(rule, model)
    follow_up_steps = 30 * max_steps
    losses = zeros(Float64, follow_up_steps)

    curr_time = time()
    p = 0.4
    steps_taken = max_steps
    for i in 1:follow_up_steps
        fill!(grads, 0.0)
        num_samples = sample_indices_(rng, p, n, indices)
        (_, current_loss) = autodiff(Enzyme.ReverseWithPrimal, component_loss, Enzyme.Active, Duplicated(model, grads), Const(radii), Const(view(indices, 1:num_samples)), Const(p))
        norm_grad = norm(grads)
        if norm_grad < 1e-9
            steps_taken = i
            break
        end
        losses[i] = current_loss
        state, model = Optimisers.update!(state, model, grads)

        if mod(i, 1000) == 0
            @info "$i iterations finished in $(time() - curr_time)s"
            curr_time = time()
        end
    end

    for i in 1:n
        for j in eachindex(positions[i])
            positions[i][j] = Tuple(collect(positions[i][j]) + model[:, i])
        end
    end

    final_positions = Vector{Tuple{Float64, Float64, Float64}}(undef, size(adjacencies, 1))
    for (i, comp) in enumerate(components)
        for (j, v) in enumerate(comp)
            final_positions[v] = positions[i][j]
        end
    end
    return final_positions, losses[1:steps_taken]
end
