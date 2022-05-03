

##########################################################
# Network regularizer
##########################################################

mutable struct NetworkRegularizer{K}

    AA::NTuple{K, <:AbstractMatrix} # Tuple of K matrices encoding relationships 
                                    # beween *observed* features

    AB::NTuple{K, <:AbstractMatrix} # Tuple of K matrices encoding relationships
                                    # between *observed and unobserved* features

    BB::NTuple{K, <:AbstractMatrix} # Tuple of K matrices encoding relationships
                                    # between *unobserved* features

    B_matrix::AbstractMatrix # a (K x N_unobserved) matrix
                             # containing estimates of 
                             # unobserved quantities
end

@functor NetworkRegularizer

function NetworkRegularizer(edgelists; observed=nothing,
                                       weight=1.0)

    # Collect all the nodes from the edgelists
    all_nodes = Set()
    for el in edgelists
        for edge in el
            push!(all_nodes, edge[1])
            push!(all_nodes, edge[2])
        end
    end

    # Figure out which nodes are observed,
    # and which are unobserved
    if observed == nothing
        observed = sort(collect(all_nodes))
        unobserved = []
    else
        # We'll just order these lexicographically
        unobserved = sort(collect(setdiff(all_nodes,observed)))
    end
    n_obs = length(observed)
    n_unobs = length(unobserved)

    # For constructing the sparse matrices, we simply
    # concatenate the observed and unobserved nodes
    allnodes_sorted = vcat(observed, unobserved)

    node_to_idx = value_to_idx(allnodes_sorted) 
    spmats = edgelists_to_spmats(edgelists, node_to_idx; epsilon=1.0)

    # Rescale the sparse matrices with the 
    # regularization weight
    for spmat in spmats
        rescale!(spmat, weight)
    end
   
    AA = Tuple(map(mat->mat[1:n_obs,1:n_obs], spmats))
    AB = Tuple(map(mat->mat[1:n_obs,n_obs+1:end], spmats))
    BB = Tuple(map(mat->mat[n_obs+1:end, n_obs+1:end], spmats))

    K = length(edgelists)
    B_matrix = randn(K, n_unobs)

    return NetworkRegularizer(AA, AB, BB, B_matrix)
end


quadratic(u::AbstractVector, 
          X::AbstractMatrix, 
          v::AbstractVector) = dot(u, (X*v))

quadratic(X::AbstractMatrix, v::AbstractVector) = dot(v, (X*v))


#################################################
# Matrix row-regularizer
#################################################

function (nr::NetworkRegularizer)(X::AbstractMatrix)

    loss = 0.0
    K = size(X,1)
    for k=1:K
        loss += quadratic(nr.AA[k], X[k,:])
        loss += 2*quadratic(X[k,:], nr.AB[k], nr.B_matrix[k,:])
        loss += quadratic(nr.BB[k], nr.B_matrix[k,:])
    end
    return 0.5*loss
end


function ChainRules.rrule(nr::NetworkRegularizer, X::AbstractMatrix)

    K, MA = size(X)
    _, MB = size(nr.B_matrix)

    # Pre-allocate these matrix-vector products
    xAA = similar(X, K,MA)
    xAB = similar(X, K,MB)
    ABu = similar(nr.B_matrix, MA,K)
    BBu = similar(nr.B_matrix, MB,K)

    loss = 0.0
    for k=1:K
        # Parts of the gradients
        xAA[k,:] = transpose(X[k,:])*nr.AA[k]
        xAB[k,:] = transpose(X[k,:])*nr.AB[k]
        ABu[:,k] = nr.AB[k]*nr.B_matrix[k,:]
        BBu[:,k] = nr.BB[k]*nr.B_matrix[k,:]

        # Full loss computation
        loss += 0.5*dot(xAA[k,:], X[k,:])
        loss += dot(X[k,:], ABu[:,k])
        loss += 0.5*dot(nr.B_matrix[k,:], BBu[:,k])
    end

    function netreg_mat_pullback(loss_bar)

        B_bar = xAB .+ transpose(BBu)
        nr_bar = Tangent{NetworkRegularizer}(B_matrix=B_bar)

        return nr_bar, xAA .+ transpose(ABu)
    end

    return loss, netreg_mat_pullback
end


#################################################
# Vector regularizer
#################################################

function (nr::NetworkRegularizer)(x::AbstractVector)

    loss = 0.0
    loss += quadratic(nr.AA[1], x)
    loss += 2*quadratic(x, nr.AB[1], nr.B_matrix[1,:])
    loss += quadratic(nr.BB[1], nr.B_matrix[1,:])
    return 0.5*loss
end


function ChainRules.rrule(nr::NetworkRegularizer, x::AbstractVector)
    loss = 0.0

    xAA = transpose(x)*nr.AA[1]
    xAB = transpose(x)*nr.AB[1]
    ABu = nr.AB[1]*nr.B_matrix[1,:]
    BBu = nr.BB[1]*nr.B_matrix[1,:]

    loss = 0.5*(dot(xAA, x) + 2*dot(x, ABu) + dot(nr.B_matrix[1,:], BBu))

    function netreg_vec_pullback(loss_bar)

        b_matrix_bar = zero(nr.B_matrix)
        b_matrix_bar[1,:] .= vec(xAB) .+ BBu

        x_bar = similar(x)
        x_bar[:] .= vec(xAA) .+ ABu

        nr_bar = Tangent{NetworkRegularizer}(B_matrix=b_matrix_bar)

        return nr_bar, x_bar
    end
    
    return loss, netreg_vec_pullback
end


##########################################################
# Network-L1 regularizer
##########################################################

mutable struct NetworkL1Regularizer{K}

    AA::NTuple{K, <:AbstractMatrix} # These tuples of matrices define 
    AB::NTuple{K, <:AbstractMatrix} # the (quadratic) network regularization.
    BB::NTuple{K, <:AbstractMatrix}

    net_virtual::NTuple{K, <:AbstractVector}
    net_weight::Number

    l1_feat_idx::NTuple{K, <:AbstractVector} # This tuple of indicator vectors
                                             # specifies which entries of the matrix
                                             # are L1-regularized.
    l1_weight::Number
end

@functor NetworkL1Regularizer


function NetworkL1Regularizer(data_features, network_edgelists;
                              net_weight=1.0, l1_weight=1.0,
                              l1_features=nothing)

    N = length(data_features)
    K = length(network_edgelists)

    AA = SparseMatrixCSC[]
    AB = SparseMatrixCSC[]
    BB = SparseMatrixCSC[]
    l1_feat_idxs = []
    virtual = Vector{Float64}[]

    datafeature_set = Set(data_features)

    # For each of the networks
    for k=1:K

        ####################################
        # Construct network regularization

        edgelist = network_edgelists[k]

        # Extract the nodes from this edgelist
        net_nodes = Set()
        for edge in edgelist
            push!(net_nodes, edge[1])
            push!(net_nodes, edge[2])
        end
        # Determine which are virtual and which are observed
        net_virtual_nodes = setdiff(net_nodes, data_features)
        net_features = setdiff(net_nodes, net_virtual_nodes)
        
        # append the virtual features to data
        net_virtual_nodes = sort(collect(net_virtual_nodes))
        all_nodes = vcat(data_features, net_virtual_nodes)
        node_to_idx = value_to_idx(all_nodes) 

        # Construct a sparse matrix encoding this network
        spmat = edgelist_to_spmat(edgelists, node_to_idx)

        # Split this matrix into observed/unobserved blocks
        push!(AA, spmat[1:N, 1:N])
        push!(AB, spmat[1:N, N+1:end])
        push!(BB, spmat[N+1:end, N+1:end])

        # Initialize a vector of virtual values
        N_virtual = length(net_virtual_nodes)
        push!(virtual, zeros(N_virtual))


        ####################################
        # Construct L1 regularization
        if l1_features == nothing
            # By default, include all non-network features
            idx_vec = map(x->!in(x, net_nodes), data_features) 
            push!(l1_feat_idxs, idx_vec) 
        else # Otherwise, include the specified features
            l1_feature_set = Set(l1_features[k])
            idx_vec = map(x->in(x, l1_feature_set), data_features)
            push!(l1_feat_idxs, idx_vec) 
        end
    end

    return NetworkL1Regularizer(Tuple(AA), 
                                Tuple(AB), 
                                Tuple(BB),
                                Tuple(virtual),
                                net_weight,
                                Tuple(l1_feat_idxs),
                                l1_weight)
end


##################################################
# Matrix row-regularization
##################################################

function (nr::NetworkL1Regularizer)(X::AbstractMatrix)

    loss = 0.0
    K = size(X,1)
    for k=1:K
        # Network-regularization
        loss += quadratic(nr.AA[k], X[k,:])
        loss += 2*quadratic(X[k,:], nr.AB[k], nr.B_matrix[k,:])
        loss += quadratic(nr.BB[k], nr.B_matrix[k,:])
        loss *= 0.5*nr.net_weight

        # L1-regularization
        loss += nr.l1_weight*sum(abs.(X[k,:].*nr.l1_feat_idxs[k]))
    end
    return loss
end


function ChainRules.rrule(nr::NetworkL1Regularizer, X::AbstractMatrix)

    K, MA = size(X)
    _, MB = size(nr.B_matrix)

    # Pre-allocate these matrix-vector products
    xAA = similar(X, K,MA)
    xAB = similar(X, K,MB)
    ABu = similar(nr.B_matrix, MA,K)
    BBu = similar(nr.B_matrix, MB,K)

    loss = 0.0
    for k=1:K

        # Network-regularization
        # Parts of the gradients
        xAA[k,:] = transpose(X[k,:])*nr.AA[k]
        xAB[k,:] = transpose(X[k,:])*nr.AB[k]
        ABu[:,k] = nr.AB[k]*nr.B_matrix[k,:]
        BBu[:,k] = nr.BB[k]*nr.B_matrix[k,:]

        # Full loss computation
        loss += 0.5*dot(xAA[k,:], X[k,:])
        loss += dot(X[k,:], ABu[:,k])
        loss += 0.5*dot(nr.B_matrix[k,:], BBu[:,k])
        loss *= nr.net_weight

        # L1-regularization
        loss += nr.l1_weight*sum(abs.(X[k,:].*nr.l1_feat_idx[k]))
    end

    function netreg_mat_pullback(loss_bar)

        B_bar = nr.net_weight.*(xAB .+ transpose(BBu))
        nr_bar = Tangent{NetworkRegularizer}(B_matrix=B_bar)

        X_bar = nr.net_weight.*(xAA .+ transpose(ABu))

        for k=1:K
            X_bar[k,:] .+= nr.l1_weight.*(sign.(X[k,:]).*nr.l1_feat_idx[k])
        end

        return nr_bar, X_bar 
    end

    return loss, netreg_mat_pullback
end


#################################################
# Vector regularizer
#################################################

function (nr::NetworkL1Regularizer)(x::AbstractVector)

    loss = 0.0
    loss += quadratic(nr.AA[1], x)
    loss += 2*quadratic(x, nr.AB[1], nr.B_matrix[1,:])
    loss += quadratic(nr.BB[1], nr.B_matrix[1,:])
    loss *= 0.5*nr.net_weight

    loss += nr.l1_weight*sum(abs.(x .* nr.l1_feat_idx[1]))

    return loss
end


function ChainRules.rrule(nr::NetworkL1Regularizer, x::AbstractVector)
    
    loss = 0.0

    xAA = transpose(x)*nr.AA[1]
    xAB = transpose(x)*nr.AB[1]
    ABu = nr.AB[1]*nr.B_matrix[1,:]
    BBu = nr.BB[1]*nr.B_matrix[1,:]

    loss = nr.net_weight*0.5*(dot(xAA, x) + 2*dot(x, ABu) + dot(nr.B_matrix[1,:], BBu))

    loss += nr.l1_weight.*sum(abs.(x .* nr.l1_feat_idx[1]))

    function netreg_vec_pullback(loss_bar)

        b_matrix_bar = zero(nr.B_matrix)
        b_matrix_bar[1,:] .= nr.net_weight.*(vec(xAB) .+ BBu)

        nr_bar = Tangent{NetworkRegularizer}(B_matrix=b_matrix_bar)
        
        x_bar = similar(x)
        x_bar[:] .= nr.net_weight.*(vec(xAA) .+ ABu)
        X_bar .+= nr.l1_weight.*(sign.(x) .* nr.l1_feat_idx[1])

        return nr_bar, x_bar
    end
    
    return loss, netreg_vec_pullback
end

#########################################
# Equality operator
#########################################

RegType = Union{NetworkRegularizer,NetworkL1Regularizer}

function Base.:(==)(a::T, b::T) where T <: RegType
    for fn in fieldnames(T)
        if !(getfield(a, fn) == getfield(b, fn)) 
            return false
        end
    end
    return true
end


