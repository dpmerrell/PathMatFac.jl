

export DEFAULT_ASSAYS, sort_features

import BatchMatFac: is_contiguous

DEFAULT_ASSAY_LOSSES = Dict("cna" => "ordinal3",
                            "mutation" => "bernoulli",
                            "methylation" => "bernoulli",
                            "mrnaseq" => "normal", 
                            "rppa" => "normal",
                            "" => "noloss"
                            )

LOSS_ORDER = Dict("normal" => 1,
                  "poisson" => 2,
                  "bernoulli" => 3,
                  "ordinal3" => 4,
                  "noloss" => 5
                  )

DOGMA_ORDER = ["dna", "mrna", "protein", "activation"]

DEFAULT_ASSAYS = collect(keys(DEFAULT_ASSAY_LOSSES))
DEFAULT_ASSAY_SET = Set(DEFAULT_ASSAYS)

DEFAULT_ASSAY_MAP = Dict("cna" => ("dna", 1),
                         "mutation" => ("dna", -1),
                         "methylation" => ("mrna", -1),
                         "mrnaseq" => ("mrna", 1),
                         "rppa" => ("protein", 1)
                        )


PWY_SIF_CODE = Dict("a" => "activation",
                    "b" => "abstract",
                    "c" => "compound",
                    "h" => "chemical",
                    "p" => "protein",
                    "f" => "family",
                    "t" => "transcription",
                    ">" => 1,
                    "|" => -1
                   )


function value_to_idx(values)
    return Dict(v => idx for (idx, v) in enumerate(values))
end


function keymatch(l_keys, r_keys)

    rkey_to_idx = value_to_idx(r_keys) 
    rkeys = Set(keys(rkey_to_idx))

    l_idx = []
    r_idx = []

    for (i, lk) in enumerate(l_keys)
        if lk in rkeys
            push!(l_idx, i)
            push!(r_idx, rkey_to_idx[lk])
        end
    end

    return l_idx, r_idx
end

function get_gene(feature)
    return feature[1]
end

function get_assay(feature)
    return feature[2] 
end


function get_loss(feature; assay_losses=DEFAULT_ASSAY_LOSSES)
    assay = get_assay(feature)
    return assay_losses[assay]
end


function srt_get_loss(feature; assay_losses=DEFAULT_ASSAY_LOSSES)
    assay = get_assay(feature)
    gene = get_gene(feature) 
    return LOSS_ORDER[assay_losses[assay]], assay, gene 
end


function sort_features(features; assay_losses=DEFAULT_ASSAY_LOSSES)
    tuple_list = [srt_get_loss(feat; assay_losses=assay_losses) for feat in features]
    sort!(tuple_list)
    return [(t[end], t[2]) for t in tuple_list]
end


function nansum(x)
    return sum(filter(!isnan, x))
end


function nanmean(x)
    return mean(filter(!isnan, x))
end


function nanvar(x)
    return var(filter(!isnan, x))
end


function edgelist_to_spmat(edgelist, node_to_idx; epsilon=0.0)

    N = length(node_to_idx)

    # make safe against redundancies.
    # in case of redundancy, keep the latest
    edge_dict = Dict()
    pwy_idx = Set()
    for edge in edgelist
        e1 = node_to_idx[edge[1]]
        e2 = node_to_idx[edge[2]]
        u = max(e1, e2)
        v = min(e1, e2)
        edge_dict[(u,v)] = edge[3]
        push!(pwy_idx, e1)
        push!(pwy_idx, e2)
    end

    # Store indices and nonzero values
    I = Int64[] 
    J = Int64[] 
    V = Float64[] 

    # Store values for the diagonal
    diagonal = zeros(N)

    # Off-diagonal entries
    for (idx, value) in edge_dict
        # below the diagonal
        push!(I, idx[1])
        push!(J, idx[2])
        push!(V, -value)
        
        # above the diagonal
        push!(I, idx[2])
        push!(J, idx[1])
        push!(V, -value)

        # increment diagonal entries
        # (maintain positive definite-ness)
        ab = abs(value)
        diagonal[idx[1]] += ab
        diagonal[idx[2]] += ab
    end
    
    for idx in pwy_idx
        diagonal[idx] += epsilon 
    end
    
    # Diagonal entries
    for i=1:N
        push!(I, i)
        push!(J, i)
        push!(V, diagonal[i])
    end

    return sparse(I, J, V)
end


function edgelists_to_spmats(edgelists, node_to_idx; epsilon=0.0)
    return [edgelist_to_spmat(el, node_to_idx; epsilon=epsilon) for el in edgelists]
end


function rescale!(spmat::SparseMatrixCSC, scalar::Number)
    spmat.nzval .*= scalar 
end


function edgelist_to_dict(edgelist)
    result = Dict()

    for edge in edgelist
        u, v, w = edge

        if haskey(result, u)
            result[u][v] = w
        else
            result[u] = Dict(v=>w)
        end
        if haskey(result, v)
            result[v][u] = w
        else
            result[v] = Dict(u=>w)
        end
    end
    return result
end


function dict_to_edgelist(graph_dict)
    result = Vector{Any}[]
    for (u,d) in graph_dict
        for (v,w) in d
            push!(result, [u,v,w])
            # Skip redundant edges
            delete!(graph_dict[v], u)
        end
    end
    return result
end


"""
    Remove all leaf nodes (degree <= 1), except those
    specified by `except`
"""
function prune_leaves!(edgelist; except=nothing)

    if except == nothing
        except_set=Set()
    else
        except_set=Set(except)
    end

    graph = edgelist_to_dict(edgelist)

    # Initialize the frontier set with the leaves
    frontier = Set([node for (node, neighbors) in graph if 
                  ((length(neighbors) < 2) & !in(node, except_set))])

    println(string("PRUNING ", length(frontier), " LEAVES"))

    # Continue until the frontier is empty
    while length(frontier) > 0
        maybe_leaf = pop!(frontier)

        # If this is, in fact, a leaf...
        if length(graph[maybe_leaf]) < 2 
            # Add the leaf's neighbor (if any) to the frontier
            for neighbor in keys(graph[maybe_leaf])
                if !in(neighbor, except_set)
                    push!(frontier, neighbor)
                end
                # Remove the leaf from the neighbor's neighbors
                delete!(graph[neighbor], maybe_leaf)
            end
            # Remove the leaf from the graph
            delete!(graph, maybe_leaf)
        end
    end

    edgelist = dict_to_edgelist(graph)
end

function get_all_nodes(edgelist)

    result = Set()
    for edge in edgelist
        push!(result, edge[1])
        push!(result, edge[2])
    end
    return result
end


