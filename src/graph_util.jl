

function graph_union(graphs::Vector)
    return sum(graphs)
end

# Returns a DFS ordering on the graph
# defined by the nonzero entries
# of a (symmetric) sparse matrix
function dfs_ordering(graph::SparseMatrixCSC; start=1)

    # Number of nodes
    N = size(graph,1)

    untouched = Set(1:N)
    counter = 1
    ordering = fill(0, N)

    stack = Int64[start]
    delete!(untouched, start)

    # Go until all nodes are visited 
    while counter <= N

        # traverse a single connected component
        while length(stack) != 0

            cur_node = pop!(stack)
            ordering[counter] = cur_node
            counter += 1

            for neighbor in graph[:,cur_node].nzind
                if neighbor in untouched
                    push!(stack, neighbor)
                    delete!(untouched, neighbor)
                end
            end
        end

        # set up to traverse the next
        # connected component
        if counter <= N
            new_start = pop!(untouched)
            stack = Int64[new_start]
            delete!(untouched, new_start)
        end
    end

    return ordering
end


# greedily assigns a "color" (integer label) to each
# node of a graph, using the given ordering on the 
# graph nodes.
function greedy_coloring(graph::SparseMatrixCSC, ordering)

    N = size(graph,1)
    coloring = fill(-1, N)

    for node in ordering
        neighbor_colors = Set( coloring[j] for j in graph[:,node].nzind )
        color = 1
        while color in neighbor_colors
            color += 1
        end
        coloring[node] = color
    end

    return coloring
end


function groupby_color(coloring::Vector{Int64})

    groups = [Vector{Int64}() for i=1:length(unique(coloring))]

    for (idx, node) in enumerate(coloring)
        push!(groups[node], idx)
    end

    return groups
end


function compute_coloring(graph::SparseMatrixCSC)

    ordering = dfs_ordering(graph)
    coloring = greedy_coloring(graph, ordering) 
    groups = groupby_color(coloring)

    return groups
end

