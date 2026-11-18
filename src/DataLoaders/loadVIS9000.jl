using DataFrames, CSV, SparseArrays

vis9000_path = "../../Data/vis9000.csv"

function extract_collaborations_vis9000(input_path::String)
    data = DataFrame(CSV.File(input_path))
    data = data[:, ["DOI", "AuthorNames", "Year"]]

    unique_authors = Set{String}()

    for row in eachrow(data)
        union!(unique_authors, split(row["AuthorNames"], ';'))
    end

    sorted_authors = sort!(collect(unique_authors))

    index_table = Dict{String,Int64}()
    for i in eachindex(sorted_authors)
        index_table[sorted_authors[i]] = i
    end

    n = length(sorted_authors)
    edges = zeros(Float64, n, n, 11)

    for row in eachrow(data)
        authors = split(row["AuthorNames"], ';')

        nauthors = length(authors)

        for i in 1:(nauthors-1)
            for j in i + 1:nauthors
                index_i = index_table[authors[i]]
                index_j = index_table[authors[j]]
                time_slice = row["Year"] - 1989

                edges[index_i, index_j, time_slice] = 1.0
                edges[index_j, index_i, time_slice] = 1.0
            end
        end
    end

    for i in 2:11
        edges[:, :, i] .= max.(edges[:, :, i-1], edges[:, :, i])
    end

    return edges
end