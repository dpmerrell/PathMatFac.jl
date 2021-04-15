

# ElUgraph represents a weighted "edge list"
# undirected graph. The edge list is symmetrized:
# (i, j, w) is in the edge list iff (j, i, w) is in the 
# edge list.
mutable struct ElUgraph
    edges::Vector
end

# Construct an ElUgraph from a (not necessarily
# symmetric) edge list
function construct_elugraph(edge_list)

    edge_dict = Dict()
    for edge in edge_list
        edge_dict[sort(edge[1:2])] = edge[3]
    end
    edges = []
    for (e, w) in edge_dict
        push!(edges, [e[1],e[2],w])
        push!(edges, [e[2],e[1],w])
    end

    return ElUgraph(edges) 
end


# LilUgraph represents a "list of lists" 
# undirected graph. The list of lists is 
# "symmetrized": node j is in node i's 
# neighbors iff node i is in node j's neighbors.
mutable struct LilUgraph
    lil::Vector{Vector{Int}}
end


# Returns a DFS ordering on the graph
# defined by `lil` (a list of lists)
function dfs_ordering(graph::LilUgraph; start=1)

    # Number of nodes
    N = length(graph.lil)

    println("\tN: ", N)

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

            for neighbor in graph.lil[cur_node]
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
function greedy_coloring(graph::LilUgraph, ordering)

    N = length(graph.lil)
    coloring = fill(-1, N)

    for node in ordering
        neighbor_colors = Set( coloring[j] for j in graph.lil[node] )
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


