


###################################################
# Math operations
###################################################

function inv_logistic(x::T) where T <: Number
    return log(0.5 + T(0.99)*(x/(1 - x) - T(0.5)))
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


###################################################
# Dealing with NaNs
###################################################
function nansum(x)
    return sum(filter(!isnan, x))
end

function nanmean(x)
    return mean(filter(!isnan, x))
end

function nanvar(x)
    return var(filter(!isnan, x))
end

function nanprint(v::Tuple, name)
    for (i,arr) in enumerate(v)
        nanprint(arr, string(name, "[", i, "]"))
    end
end

function nanprint(v::AbstractArray, name)
    s = sum((!isfinite).(v))
    if s > 0
        println(string(s, " NONFINITE VALUES IN ", name))
    end
end


######################################################
# Hardware-agnostic, GPU-enabled functions
######################################################

function randn_like(A::CuArray)
    M, N = size(A)
    return CUDA.randn(M,N)
end

function randn_like(A::Array)
    M, N = size(A)
    return randn(M,N)
end

function zeros_like(args...)
    result = similar(args...)
    result .= 0
    return result
end

function ones_like(args...)
    result = similar(args...)
    result .= 1
    return result
end

function set_array!(target::CuArray, source::AbstractArray)
    target .= gpu(source)
end

function set_array!(target::Array, source::AbstractArray)
    target .= cpu(source)
end


#####################################################
# I/O Utils
#####################################################

FIT_START_TIME = time()

function v_print(args...; verbosity=1, level=1, prefix="")
    if verbosity >= level
        print(string(prefix, args...))
    end
end

function v_println(args...; kwargs...)
    v_print(args..., " (", Int(round(time() - FIT_START_TIME)), "s elapsed)\n"; kwargs...)
end

######################################################
# Canonical central dogma relationships 
######################################################


DOGMA_ORDER = ["dna", "mrna", "protein", "activation"]
DOGMA_TO_IDX = Dict([v => i for (i,v) in enumerate(DOGMA_ORDER)])

PWY_SIF_CODE = Dict("a" => "activation",
                    "d" => "dna",
                    "t" => "mrna",
                    "p" => "protein",
                    ">" => 1,
                    "|" => -1
                   )

VALID_LOSSES = ["normal", "bernoulli", "poisson", "ordinal3"]

###############################################
# Index and ID utils
###############################################

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


# Assume ranges are nonoverlapping, and in sorted order.
function subset_ranges(ranges::Vector, rng::UnitRange) 

    if length(ranges) == 0
        return UnitRange[], 1, 0
    end  
 
    r_min = max(rng.start, ranges[1].start)
    r_max = min(rng.stop, ranges[end].stop)
    if r_min > r_max
        return UnitRange[], 1, 0 
    end
    @assert r_min <= r_max

    @assert r_min >= ranges[1].start
    @assert r_max <= ranges[end].stop

    starts = [rr.start for rr in ranges]
    r_min_idx = searchsorted(starts, r_min).stop
    if (r_min > ranges[r_min_idx].stop)
        r_min_idx += 1
        r_min = ranges[r_min_idx].start
    end

    stops = [rr.stop for rr in ranges]
    r_max_idx = searchsorted(stops, r_max).start
    if (r_max < ranges[r_max_idx].start)
        r_max_idx -= 1
        r_max = ranges[r_max_idx].stop
    end

    if r_min_idx > r_max_idx
        return UnitRange[], 1, 0
    end

    new_ranges = ranges[r_min_idx:r_max_idx]
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


################################################
# Edgelist utils
################################################

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


function scale_spmat!(spmat::SparseMatrixCSC, scalar::Number)
    spmat.nzval .*= scalar 
end

function scale_spmat!(spmat::CUDA.CUSPARSE.CuSparseMatrixCSC, scalar::Number)
    spmat.nzVal .*= scalar 
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
    frontier = Set(node for (node, neighbors) in graph if 
                   ((length(neighbors) < 2) & !in(node, except_set))
                  )

    # Continue until the frontier is empty
    while length(frontier) > 0
        maybe_leaf = pop!(frontier)

        # If this is, in fact, a leaf...
        if length(graph[maybe_leaf]) < 2 
            # Add the leaf's neighbor (if any) to the frontier
            for neighbor in keys(graph[maybe_leaf])
                if !in(neighbor, except_set) & (neighbor != maybe_leaf)
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


function get_all_entities(edgelist)

    result = Set()
    for edge in edgelist
        u = split(edge[1], "_")[1]
        v = split(edge[2], "_")[1]
        push!(result, u)
        push!(result, v)
    end
    return result
end


"""
    For each edgelist in `edgelist`, find the 
    nodes in `nodes` that do not appear in it.
    (I.e., a set difference) 
"""
function compute_nongraph_nodes(nodes, edgelists)

    all_nodes = Set(nodes)
    result = []
    for el in edgelists
        edgelist_nodes = get_all_nodes(el)
        push!(result, setdiff(all_nodes, edgelist_nodes))
    end

    return result
end


################################################
# Sparse matrix utils
################################################

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

function construct_row_selector(rs::SparseMatrixCSC, idx::AbstractVector{Int})
    M = length(idx)
    N = rs.m
    I = collect(1:M)
    J = idx
    V = ones(Bool, M)
    return sparse(I, J, V, M, N)
end

function construct_row_selector(rs::SparseMatrixCSC, idx::UnitRange)
    M = length(idx)
    N = rs.m
    I = collect(1:M)
    J = collect(idx)
    V = ones(Bool, M)
    return sparse(I, J, V, M, N)
end


function construct_row_selector(rs::CUDA.CUSPARSE.CuSparseMatrixCSC, idx::AbstractVector{Int})

    M = length(idx)
    N = rs.dims[1]
    I = similar(rs.rowVal, M)
    I .= 1
    I .= cumsum(I)
    J = cu(collect(idx))
    V = CUDA.ones(M)

    coo = CUDA.CUSPARSE.CuSparseMatrixCOO(I,J,V,(M,N),M)
    return CUDA.CUSPARSE.CuSparseMatrixCSC(coo)
end

function construct_row_selector(rs::CUDA.CUSPARSE.CuSparseMatrixCSC, idx::CuVector)

    M = length(idx)
    N = rs.dims[1]
    I = similar(rs.rowVal, M)
    I .= 1
    I .= cumsum(I)
    J = idx
    V = CUDA.ones(M)

    coo = CUDA.CUSPARSE.CuSparseMatrixCOO(I,J,V,(M,N),M)
    return CUDA.CUSPARSE.CuSparseMatrixCSC(coo)
end

function construct_row_selector(rs::CUDA.CUSPARSE.CuSparseMatrixCSC, idx::UnitRange)

    M = length(idx)
    N = rs.dims[1]
    I = similar(rs.rowVal, M)
    I .= 1
    I .= cumsum(I)
    J = cu(collect(idx))
    V = CUDA.ones(M)

    coo = CUDA.CUSPARSE.CuSparseMatrixCOO(I,J,V,(M,N),M)
    return CUDA.CUSPARSE.CuSparseMatrixCSC(coo)
end

# `select_rows` wraps Boolean matrix multiplication
# and handles some weirdness specific to CUDA sparse matrices.
function select_rows(rs::SparseMatrixCSC, rb::SparseMatrixCSC)
    return rs * rb
end

function select_rows(rs::CUDA.CUSPARSE.CuSparseMatrixCSC, rb::CUDA.CUSPARSE.CuSparseMatrixCSC)
    result = rs * rb
    result.nzVal .= 1
    return result
end


# Get the indices of the entries for column j in a CSC sparse matrix
function get_col_idx(sp::SparseMatrixCSC, j)
    idx = sp.colptr[j:(j+1)]
    start = idx[1]
    stop = idx[2]-1
    return sp.rowval[start:stop]
end

function get_col_idx(sp::CUDA.CUSPARSE.CuSparseMatrixCSC, j)
    idx = cpu(sp.colPtr[j:(j+1)])
    start = idx[1]
    stop = idx[2] - 1
    return sp.rowVal[start:stop]
end

#########################################################
# History utils
#########################################################


function history!(hist; kwargs...)
    d = Dict()
    for k in keys(kwargs)
        d[string(k)] = kwargs[k]
    end
    d["time"] = time() - FIT_START_TIME
    push!(hist, d)
end

function history!(hist, d::AbstractDict; kwargs...)
    for k in keys(kwargs)
        d[string(k)] = kwargs[k]
    end
    d["time"] = time()
    push!(hist, d)
end

function finalize_history(hist)
    return collect(hist)
end

function write_history(hist, json_path)
    open(json_path, "w") do f
        JSON.print(f, hist)
    end
end


function history!(::Nothing; kwargs...)
    return
end

function history!(::Nothing, d::Any; kwargs...)
    return
end

function finalize_history(::Nothing)
    return
end

function write_history(hist::Nothing, json_path)
    return
end

#########################################################
# Other
#########################################################

# Binary search on a "well-behaved" nondecreasing function, f.
# I.e., given a target value z, search for  x satisfying 
#      f(x) = z
# or, more accurately,
#      |f(x) - z| < z_atol
# We assume f is a function of **positive real numbers**.
#
# It may also terminate if it finds lower and upper bounds
# satisfying (UB - LB) < x_atol.
function func_binary_search(x_start, z_target, f; z_atol=1e-1, x_atol=1e-3, max_iter=20, 
                                                  verbosity=1, print_prefix="")
    UB = Inf
    LB = -Inf
    x = x_start
    y = f(x)
    iter = 0
    # Obtain finite LB and UB.
    # If f(x) too small, keep doubling x until we have a finite UB.
    sz = sign(z_target)
    if y < z_target - z_atol
        while (y < z_target) & (iter < max_iter)
            LB = x
            x *= 2
            y = f(x)
            iter += 1
        end
        UB = x
    # If f(x) too big, keep halving x until we have a finite LB.
    elseif y > z_target + z_atol
        while (y > z_target) & (iter < max_iter)
            UB = x
            x *= 0.5
            y = f(x)
            iter += 1
        end
        LB = x
    end
    # Check whether we hit the target during the bounding phase
    if abs(y - z_target) < z_atol 
        return x, y
    end

    # Now that we have finite LB and UB, search between them
    # with standard binary search
    while iter < max_iter
        delta = UB - LB
        if delta < x_atol
            break
        end
        x = 0.5*(LB + UB)
        y = f(x)
        if y < z_target - z_atol
            LB = x
        elseif y > z_target + z_atol
            UB = x
        else
            break
        end 
        iter += 1
    end
    
    if iter >= max_iter
        v_println("WARNING: binary search hit max_iter=", max_iter; verbosity=verbosity,
                                                                    prefix=print_prefix)
    end

    return x, y
end


