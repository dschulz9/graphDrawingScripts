using GLMakie, GraphMakie, Graphs

include("example_graphs.jl")
include("MPSE_autodiff.jl")

curr_title = "VIS900"

D = hexa_square
labels = ["$i" for i in axes(D, 1)]
n, _, K = size(D)
steps = 5_000
seed = 773474612908383872
tuples, views, losses, sgd_loss, X, U = calculate_positions(D, steps, 0.4, seed)
tuple_layout(_) = tuples

f, ax, p = graphplot(SimpleGraph(D[:, :, end]); layout=tuple_layout, nlabels=labels)
save("figures/$(curr_title)_3d.png", f)

for i in 1:K
    f, ax, p = graphplot(SimpleGraph(D[:, :, i]), layout=views[i], nlabels=labels)
    save("figures/$(curr_title)_step_$i.png", f)
end

fig = Figure()
ax = Axis(fig[1, 1]; yscale=log10)
# ax = Axis(fig[1,1])

lines!(ax, 1:length(sgd_loss), sgd_loss, label="Stabilized SGD loss")
lines!(ax, 1:length(losses[1:end]), losses[1:end], label="MDS loss")
axislegend(ax)
save("figures/$(curr_title)_loss.png",fig)

# tetrahedron = [0.0 0.5 1 0.5; 0.0 sqrt(3)/2 0 1/(2 * sqrt(3)); 0 0 0 sqrt(2/3)]
# tuples_tetrahedron = [Tuple(tetrahedron[:, k]) for k in axes(tetrahedron, 2)]
# projected_tuples_tetrahedron = [Tuple(tetrahedron[1:2, k]) for k in axes(tetrahedron, 2)]
# projections_tetrahedron = [0 0; 0 0; 1.0 1.0]
# D_tetrahedron = fully_connected_4_build_up[:, :, 2:3]
# loss(D_tetrahedron, tetrahedron, projections_tetrahedron, sum(D_tetrahedron .^ 2))
# graphplot(SimpleGraph(D_tetrahedron[:, :, end-1]), layout=projected_tuples_tetrahedron)
# graphplot(SimpleGraph(D_tetrahedron[:, :, end]), layout=projected_tuples_tetrahedron)

include("DataLoaders/loadVIS9000.jl")

collaborations = extract_collaborations_vis9000(vis9000_path)
@info any(isnan, collaborations)
for i in axes(collaborations, 3)
    @info all(iszero, diag(collaborations[:, :, i]))
end

tuples, losses = place_unconnected_graph(collaborations, steps, 0.3, seed)
tuple_layout(_) = tuples
labels = ["$i" for i in axes(collaborations, 1)]
f, ax, p = graphplot(SimpleGraph(collaborations[:, :, end]); layout=tuple_layout, nlabels=labels)
save("figures/VIS9000_3d.png", f)
graphplot(SimpleGraph(collaborations[:, :, 1]), layout=views[1], nlabels=labels)

fc = generate_fully_connected(10, 4)
labels = ["$i" for i in axes(fc, 1)]
tuples, views, losses, sgd_loss, X, U = calculate_positions(fc, steps, 0.4, 778383872)
tuple_layout(_) = tuples
f, ax, p = graphplot(SimpleGraph(fc[:, :, end]); layout=tuple_layout, nlabels=labels)
graphplot(SimpleGraph(fc[:, :, 1]), layout=views[1], nlabels=labels)
graphplot(SimpleGraph(fc[:, :, 2]), layout=views[2], nlabels=labels)
graphplot(SimpleGraph(fc[:, :, 3]), layout=views[3], nlabels=labels)
graphplot(SimpleGraph(fc[:, :, 4]), layout=views[4], nlabels=labels)
fig = Figure()
ax = Axis(fig[1, 1]; yscale=log10)
lines!(ax, 1:length(losses[1:end-1]), losses[1:end-1], label="MDS loss")