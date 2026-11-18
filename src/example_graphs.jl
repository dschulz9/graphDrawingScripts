fully_connected_4_build_up = zeros(Float64, 4, 4, 3)
fully_connected_4_build_up[:, :, 1] = [0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0]
fully_connected_4_build_up[:, :, 2] = [0 1 1 0; 1 0 1 0; 1 1 0 0; 0 0 0 0]
fully_connected_4_build_up[:, :, 3] = [0 1 1 1; 1 0 1 1; 1 1 0 1; 1 1 1 0]

fully_connected_4_build_up_exist = BitArray([1 1 1; 0 1 1; 0 1 1; 0 0 1])

# Try MLDataSets: ChickenPox

line_to_square = zeros(Float64, 6, 6, 4)
line_to_square[:, :, 1] = [0 0 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0]
line_to_square[:, :, 2] = [0 1 0 0 0 0; 1 0 1 0 0 0; 0 1 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0]
line_to_square[:, :, 3] = [0 1 0 1 0 0; 1 0 1 0 0 0; 0 1 0 1 0 0; 1 0 1 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0]
line_to_square[:, :, 4] = [0 1 0 1 1 0; 1 0 1 0 0 1; 0 1 0 1 0 0; 1 0 1 0 0 0; 1 0 0 0 0 1; 0 1 0 0 1 0]

hexa_square = zeros(Float64, 10, 10, 4)
hexa_square[:, :, 1] = [
    0 1 0 0 0 0 0 0 0 0;
    1 0 1 0 0 0 0 0 0 0;
    0 1 0 1 0 0 0 0 0 0;
    0 0 1 0 1 0 0 0 0 0;
    0 0 0 1 0 1 0 0 0 0;
    0 0 0 0 1 0 0 0 0 0;
    0 0 0 0 0 0 0 0 0 0;
    0 0 0 0 0 0 0 0 0 0;
    0 0 0 0 0 0 0 0 0 0;
    0 0 0 0 0 0 0 0 0 0;
]
hexa_square[:, :, 2] = [
    0 1 0 0 1 1 0 0 0 0;
    1 0 1 1 0 0 0 0 0 0;
    0 1 0 1 0 0 0 0 0 0;
    0 1 1 0 1 0 0 0 0 0;
    1 0 0 1 0 1 0 0 0 0;
    1 0 0 0 1 0 0 0 0 0;
    0 0 0 0 0 0 0 0 0 0;
    0 0 0 0 0 0 0 0 0 0;
    0 0 0 0 0 0 0 0 0 0;
    0 0 0 0 0 0 0 0 0 0;
]
hexa_square[:, :, 3] = [
    0 1 0 0 1 1 1 0 0 0;
    1 0 1 1 0 0 0 1 0 0;
    0 1 0 1 0 0 0 0 0 0;
    0 1 1 0 1 0 0 0 0 0;
    1 0 0 1 0 1 0 0 0 0;
    1 0 0 0 1 0 0 0 0 0;
    1 0 0 0 0 0 0 1 0 0;
    0 1 0 0 0 0 1 0 0 0;
    0 0 0 0 0 0 0 0 0 0;
    0 0 0 0 0 0 0 0 0 0;
]

hexa_square[:, :, 4] = [
    0 1 0 0 1 1 1 0 0 1;
    1 0 1 1 0 0 0 1 0 1;
    0 1 0 1 0 0 0 0 0 0;
    0 1 1 0 1 0 0 0 0 0;
    1 0 0 1 0 1 0 0 0 0;
    1 0 0 0 1 0 0 0 0 0;
    1 0 0 0 0 0 0 1 0 1;
    0 1 0 0 0 0 1 0 0 1;
    0 0 0 0 0 0 0 0 0 1;
    1 1 0 0 0 0 1 1 1 0;
]

function generate_fully_connected(nodes::Int64, steps::Int64)
    g = SimpleGraph()
    add_vertices!(g, nodes)

    adjacencies = zeros(Float64, nodes, nodes, steps)

    chunk_size = floor(nodes/steps)
    for k in 1:steps
        range_start = (k - 1) * chunk_size

        for i in 1:chunk_size
            my_index = range_start + i
            for j in 1:my_index - 1
                add_edge!(g, j, my_index)
            end
        end

        adjacencies[:, :, k] = adjacency_matrix(g)
    end

    return adjacencies
end