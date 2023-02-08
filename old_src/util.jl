

export DEFAULT_ASSAYS, sort_features


DEFAULT_ASSAY_LOSSES = Dict("cna" => "ordinal3",
                            "mutation" => "bernoulli",
                            "methylation" => "bernoulli",
                            "mrnaseq" => "normal", 
                            "rppa" => "normal",
                            )

LOSS_ORDER = Dict("normal" => 1,
                  "poisson" => 2,
                  "bernoulli" => 3,
                  "ordinal3" => 4,
                  "noloss" => 5
                  )

function inv_logistic(x::T) where T <: Number
    return max(log(x / (1 - x)), T(-10.0))
end

function bernoulli_var_init(m::T, v::T) where T <: Number
    if (m == 0) | (m==1)
        return T(0)
    else
        denom = m*(1-m)
        denom *= denom
        var = v / denom
        var = min(var, T(9.0))
        return var
    end
end

MEAN_INIT_MAP = Dict("normal" => (m,v) -> m,
                     "bernoulli" => (m,v) -> inv_logistic(m),
                     "ordinal3" => (m,v) -> m - 2.0,
                     "poisson" => (m,v) -> log(m)
                    )
VAR_INIT_MAP = Dict("normal" => (m,v) -> v,
                    "bernoulli" => (m,v) -> bernoulli_var_init(m,v),
                    "ordinal3" => (m,v) -> v,
                    "poisson" => (m,v) -> v .* (log.(m).^2)
                   )


DOGMA_ORDER = ["dna", "mrna", "protein", "activation"]
DOGMA_TO_IDX = Dict([v => i for (i,v) in enumerate(DOGMA_ORDER)])

DEFAULT_ASSAYS = collect(keys(DEFAULT_ASSAY_LOSSES))
DEFAULT_ASSAY_SET = Set(DEFAULT_ASSAYS)

DEFAULT_ASSAY_MAP = Dict("cna" => ("dna", 1),
                         "mutation" => ("dna", -1),
                         "methylation" => ("mrna", -1),
                         "mrnaseq" => ("mrna", 1),
                         "rppa" => ("protein", 1)
                        )


PWY_SIF_CODE = Dict("a" => "activation",
                    "d" => "dna",
                    "t" => "mrna",
                    "p" => "protein",
                    ">" => 1,
                    "|" => -1
                   )


"""
Check that the values in `vec` occur in contiguous blocks.
I.e., the unique values are grouped together, with no intermingling.
I.e., for each unique value the set of indices mapping to that value
occur consecutively.
"""
function is_contiguous(vec::AbstractVector{T}) where T

    past_values = Set{T}()
    
    for i=1:(length(vec)-1)
        next_value = vec[i+1]
        if in(vec[i+1], past_values)
            return false
        end
        if vec[i+1] != vec[i]
            push!(past_values, vec[i])
        end
    end

    return true
end


function ids_to_ranges(id_vec)

    @assert is_contiguous(id_vec) "IDs in id_vec need to appear in contiguous chunks."

    unique_ids = unique(id_vec)
    start_idx = indexin(unique_ids, id_vec)
    end_idx = length(id_vec) .- indexin(unique_ids, reverse(id_vec)) .+ 1
    ranges = UnitRange[start:finish for (start,finish) in zip(start_idx, end_idx)]

    return ranges
end


function ids_to_ind_mat(id_vec)

    unq_ids = unique(id_vec)
    ind_mat = zeros(Bool, length(id_vec), length(unq_ids))

    for (i,name) in enumerate(unq_ids)
        ind_mat[:,i] .= (id_vec .== name)
    end

    return ind_mat
end


function subset_ranges(ranges::Vector, rng::UnitRange) 
   
    r_min = rng.start
    r_max = rng.stop
    @assert r_min <= r_max

    @assert r_min >= ranges[1].start
    @assert r_max <= ranges[end].stop

    starts = [rr.start for rr in ranges]
    r_min_idx = searchsorted(starts, r_min).stop
    
    stops = [rr.stop for rr in ranges]
    r_max_idx = searchsorted(stops, r_max).start

    new_ranges = collect(ranges[r_min_idx:r_max_idx])
    new_ranges[1] = r_min:new_ranges[1].stop
    new_ranges[end] = new_ranges[end].start:r_max

    return new_ranges, r_min_idx, r_max_idx
end

function subset_ranges(ranges::Tuple, rng::UnitRange)
    new_ranges, r_min, r_max = subset_ranges(collect(ranges), rng)
    return Tuple(new_ranges), r_min, r_max
end

function shift_range(rng, delta)
    return (rng.start + delta):(rng.stop + delta) 
end

function value_to_idx(values::Vector{T}) where T
    d = Dict{T,Int64}()
    sizehint!(d, length(values))
    for (i,v) in enumerate(values)
        d[v] = i
    end
    return d
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
    sizehint!(edge_dict, length(edgelist))
    for edge in edgelist
        e1 = node_to_idx[edge[1]]
        e2 = node_to_idx[edge[2]]
        i = max(e1, e2)
        j = min(e1, e2)
        edge_dict[(i,j)] = edge[3]
    end
    enc_edgelist = [(i,j,v) for ((i,j),v) in edge_dict]

    # Store indices and nonzero values
    n_edges = length(enc_edgelist)
    mid = N + n_edges
    nnz = N + 2*n_edges
    I = zeros(Int64, nnz) 
    J = zeros(Int64, nnz)
    V = zeros(Float64, nnz)
    
    # Initialize diagonal entries:
    I[1:N] .= 1:N
    J[1:N] .= 1:N
    V[1:N] .= epsilon

    for (count, (i,j,v)) in enumerate(enc_edgelist)
        I[N+count] = i
        J[N+count] = j
        V[N+count] = -v

        I[mid+count] = j
        J[mid+count] = i
        V[mid+count] = -v

        av = abs(v)
        V[i] += av
        V[j] += av
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


function csc_to_coo(A)
    I = zero(A.rowval)
    J = zero(A.rowval)
    V = zero(A.nzval)

    for j=1:A.n
        cpt = (A.colptr[j]):(A.colptr[j+1]-1)
        J[cpt] .= j
        I[cpt] .= A.rowval[cpt]
        V[cpt] .= A.nzval[cpt]
    end

    return I, J, V
end

function coo_select(I,J,V, rng1::UnitRange, rng2::UnitRange)

    keep_idx = (((I .>= rng1.start) .& (I .<= rng1.stop)) .& (J .>= rng2.start)) .& (J .<= rng2.stop) 

    I_new = I[keep_idx] .- (rng1.start - 1)
    J_new = J[keep_idx] .- (rng2.start - 1)
    V_new = V[keep_idx]

    return I_new, J_new, V_new
end

function csc_select(A, rng1::UnitRange, rng2::UnitRange)

    new_m = length(rng1)
    new_n = length(rng2)
    return sparse(coo_select(csc_to_coo(A)..., rng1, rng2)..., new_m, new_n) 
end


