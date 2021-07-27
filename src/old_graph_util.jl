

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


