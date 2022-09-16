
import Base: ==


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
Flux.trainable(nr::NetworkRegularizer) = (B_matrix=nr.B_matrix, )

function NetworkRegularizer(edgelists; observed=nothing,
                                       weight=1.0,
                                       epsilon=0.0)

    # Collect all the nodes from the edgelists
    all_nodes = Set()
    for el in edgelists
        union!(all_nodes, get_all_nodes(el))
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
    n_total = n_obs + n_unobs

    # For constructing the sparse matrices, we simply
    # concatenate the observed and unobserved nodes
    allnodes_sorted = vcat(observed, unobserved)

    node_to_idx = value_to_idx(allnodes_sorted) 
    spmats = edgelists_to_spmats(edgelists, node_to_idx; epsilon=epsilon)

    # Rescale the sparse matrices with the 
    # regularization weight
    for spmat in spmats
        rescale!(spmat, weight)
    end
    AA = Tuple(map(mat->csc_select(mat, 1:n_obs, 1:n_obs), spmats))
    AB = Tuple(map(mat->csc_select(mat, 1:n_obs, n_obs+1:n_total), spmats))
    BB = Tuple(map(mat->csc_select(mat, n_obs+1:n_total, n_obs+1:n_total), spmats))

    K = length(edgelists)
    B_matrix = zeros(K, n_unobs)

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
    K,M = size(X)
    for k=1:K
        loss += quadratic(nr.AA[k], X[k,:])
        loss += 2*quadratic(X[k,:], nr.AB[k], nr.B_matrix[k,:])
        loss += quadratic(nr.BB[k], nr.B_matrix[k,:])
    end
    return 0.5*loss/(K*M)
end


function ChainRules.rrule(nr::NetworkRegularizer, X::AbstractMatrix)

    K, MA = size(X)
    _, MB = size(nr.B_matrix)
    inv_KM = 1/(K*MA)

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
    loss *= inv_KM

    function netreg_mat_pullback(loss_bar)

        B_bar = loss_bar.*(xAB .+ transpose(BBu)) .* inv_KM 
        nr_bar = Tangent{NetworkRegularizer}(B_matrix=B_bar) 

        return nr_bar, loss_bar .* (xAA .+ transpose(ABu)) .* inv_KM
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
    return 0.5*loss ./length(x)
end


function ChainRules.rrule(nr::NetworkRegularizer, x::AbstractVector)
    loss = 0.0
    inv_M = 1/length(x)
    xAA = transpose(x)*nr.AA[1]
    xAB = transpose(x)*nr.AB[1]
    ABu = nr.AB[1]*nr.B_matrix[1,:]
    BBu = nr.BB[1]*nr.B_matrix[1,:]

    loss = inv_M*0.5*(dot(xAA, x) + 2*dot(x, ABu) + dot(nr.B_matrix[1,:], BBu))

    function netreg_vec_pullback(loss_bar)

        b_matrix_bar = zero(nr.B_matrix)
        b_matrix_bar[1,:] .= inv_M .* loss_bar.*(vec(xAB) .+ BBu) 

        x_bar = similar(x)
        x_bar[:] .= inv_M .* loss_bar.*(vec(xAA) .+ ABu)

        nr_bar = Tangent{NetworkRegularizer}(B_matrix=b_matrix_bar)

        return nr_bar, x_bar
    end
    
    return loss, netreg_vec_pullback
end

##########################################################
# Cluster Regularizer
##########################################################

mutable struct ClusterRegularizer 
    cluster_idx::AbstractMatrix{Float32}
    cluster_sizes::AbstractVector{Int32}
    weight::Number
end

@functor ClusterRegularizer
Flux.trainable(cr::ClusterRegularizer) = ()

function ClusterRegularizer(cluster_labels::AbstractVector; weight=1.0)
    idx = ids_to_ind_mat(cluster_labels)
    sizes = vec(sum(idx, dims=1))
    idx = sparse(idx)
    return ClusterRegularizer(idx, sizes, weight)
end


#####################################
# regularize a vector
function (cr::ClusterRegularizer)(x::AbstractVector)

    means = (transpose(cr.cluster_idx) * x) ./ cr.cluster_sizes
    mean_vec = cr.cluster_idx * means
    diffs = x .- mean_vec
    return 0.5*cr.weight*sum(diffs.*diffs) / length(x)
end


function ChainRulesCore.rrule(cr::ClusterRegularizer, x::AbstractVector)
   
    means = (transpose(cr.cluster_idx) * x) ./ cr.cluster_sizes
    mean_vec = cr.cluster_idx * means
    diffs = x .- mean_vec
    inv_len = 1 / length(x)

    function cluster_reg_pullback(loss_bar)
        return ChainRulesCore.NoTangent(), 
               (cr.weight*inv_len*loss_bar) .* diffs 
    end
    loss = 0.5*cr.weight*inv_len*sum(diffs.*diffs)

    return loss, cluster_reg_pullback
end


####################################
# Regularize rows of a matrix
function (cr::ClusterRegularizer)(X::AbstractMatrix)

    means = (transpose(cr.cluster_idx) * transpose(X)) ./ cr.cluster_sizes
    mean_rows = transpose(cr.cluster_idx * means)
    diffs = X .- mean_rows

    return (0.5*cr.weight/prod(size(X)))*sum(diffs.*diffs) 
end


function ChainRulesCore.rrule(cr::ClusterRegularizer, X::AbstractMatrix)

    means = (transpose(cr.cluster_idx) * transpose(X)) ./ cr.cluster_sizes
    mean_rows = transpose(cr.cluster_idx * means)
    diffs = X .- mean_rows

    inv_size = 1/prod(size(X))
    function cluster_reg_pullback(loss_bar)
        return ChainRulesCore.NoTangent(), 
               (cr.weight*loss_bar*inv_size) .* diffs
    end
    loss = 0.5*cr.weight*inv_size*sum(diffs.*diffs)

    return loss, cluster_reg_pullback
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
Flux.trainable(nr::NetworkL1Regularizer) = (net_virtual=nr.net_virtual,)


function NetworkL1Regularizer(data_features::Vector, network_edgelists::Vector;
                              net_weight=1.0, l1_weight=1.0,
                              l1_features=nothing,
                              epsilon=1.0)

    N = length(data_features)
    K = length(network_edgelists)

    AA = Vector{SparseMatrixCSC}(undef, K) 
    AB = Vector{SparseMatrixCSC}(undef, K)
    BB = Vector{SparseMatrixCSC}(undef, K)
    l1_feat_idxs = Vector{Vector{Bool}}(undef, K)
    virtual = Vector{Float64}[]

    datafeature_set = Set(data_features)

    # For each of the networks
    for k=1:K

        ####################################
        # Construct network regularization
        edgelist = network_edgelists[k]

        # Extract the nodes from this edgelist
        net_nodes = get_all_nodes(edgelist)

        # Determine which are virtual and which are observed
        net_virtual_nodes = setdiff(net_nodes, data_features)
        net_features = setdiff(net_nodes, net_virtual_nodes)
        
        # append the virtual features to data
        net_virtual_nodes = sort(collect(net_virtual_nodes))
        all_nodes = vcat(data_features, net_virtual_nodes)
        node_to_idx = value_to_idx(all_nodes) 

        # Construct a sparse matrix encoding this network
        spmat = edgelist_to_spmat(edgelist, node_to_idx; epsilon=epsilon)

        # Split this matrix into observed/unobserved blocks
        N_total = size(spmat, 2)
        AA[k] = csc_select(spmat, 1:N, 1:N)
        AB[k] = csc_select(spmat, 1:N, N+1:N_total)
        BB[k] = csc_select(spmat, N+1:N_total, N+1:N_total)

        # Initialize a vector of virtual values
        N_virtual = length(net_virtual_nodes)
        push!(virtual, zeros(N_virtual))

        ####################################
        # Construct L1 regularization
        if l1_features == nothing
            # By default, include all non-network features
            idx_vec = map(x->!in(x, net_nodes), data_features) 
            l1_feat_idxs[k] = idx_vec
        else # Otherwise, include the specified features
            l1_feature_set = Set(l1_features[k])
            idx_vec = map(x->in(x, l1_feature_set), data_features)
            l1_feat_idxs[k] = idx_vec
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
    K,M = size(X)
    for k=1:K
        # Network-regularization
        net_loss = 0.0
        net_loss += quadratic(nr.AA[k], X[k,:])
        net_loss += 2*quadratic(X[k,:], nr.AB[k], nr.net_virtual[k])
        net_loss += quadratic(nr.BB[k], nr.net_virtual[k])
        net_loss *= 0.5*nr.net_weight
        loss += net_loss

        # L1-regularization
        loss += nr.l1_weight*sum(abs.(X[k,:].*nr.l1_feat_idxs[k]))
    end
    loss /= (K*M)
    return loss
end


function ChainRules.rrule(nr::NetworkL1Regularizer, X::AbstractMatrix)

    K, MA = size(X)
    inv_KM = 1/(K*MA)

    # Pre-allocate these matrix-vector products
    xAA = similar(X, K,MA)
    xAB = map(similar, nr.net_virtual)
    ABu = similar(X, MA,K)
    BBu = map(similar, nr.net_virtual)

    loss = 0.0
    for k=1:K

        # Network-regularization
        # Parts of the gradients
        xAA[k,:] .= vec(transpose(X[k,:])*nr.AA[k])
        xAB[k] .= vec(transpose(X[k,:])*nr.AB[k])
        ABu[:,k] .= nr.AB[k]*nr.net_virtual[k]
        BBu[k] .= nr.BB[k]*nr.net_virtual[k]

        # Full loss computation
        net_loss = 0.0
        net_loss += 0.5*dot(xAA[k,:], X[k,:])
        net_loss += dot(X[k,:], ABu[:,k])
        net_loss += 0.5*dot(nr.net_virtual[k], BBu[k])
        net_loss *= nr.net_weight
        loss += net_loss 

        # L1-regularization
        loss += nr.l1_weight * sum(abs.(X[k,:]) .* nr.l1_feat_idx[k]) 
    end
    loss *= inv_KM

    function netreg_mat_pullback(loss_bar)

        # Gradient w.r.t. the network's virtual nodes
        virt_bar = map(similar, nr.net_virtual)
        for k=1:K
            virt_bar[k] .= (loss_bar*nr.net_weight*inv_KM).*(xAB[k] .+ BBu[k])
        end
        nr_bar = Tangent{NetworkL1Regularizer}(net_virtual=virt_bar)

        # Network regularization for the observed nodes
        X_bar = nr.net_weight.*(xAA .+ transpose(ABu))
        # L1 regularization for the observed nodes
        for k=1:K
            X_bar[k,:] .+= nr.l1_weight.*(sign.(X[k,:]).*nr.l1_feat_idx[k])
        end
        X_bar .*= loss_bar
        X_bar .*= inv_KM

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
    loss += 2*quadratic(x, nr.AB[1], nr.net_virtual[1])
    loss += quadratic(nr.BB[1], nr.net_virtual[1])
    loss *= 0.5*nr.net_weight

    loss += nr.l1_weight*sum(abs.(x .* nr.l1_feat_idx[1]))
    loss /= length(x)

    return loss
end


function ChainRules.rrule(nr::NetworkL1Regularizer, x::AbstractVector)
    
    loss = 0.0
    inv_M = length(x)

    xAA = transpose(x)*nr.AA[1]
    xAB = transpose(x)*nr.AB[1]
    ABu = nr.AB[1]*nr.net_virtual[1]
    BBu = nr.BB[1]*nr.net_virtual[1]

    loss = nr.net_weight*0.5*(dot(xAA, x) + 2*dot(x, ABu) + dot(nr.net_virtual[1], BBu))

    loss += nr.l1_weight.*sum(abs.(x .* nr.l1_feat_idx[1]))
    loss .*= inv_M 

    function netreg_vec_pullback(loss_bar)

        virt_bar = map(zero, nr.net_virtual)
        virt_bar[1] .= (loss_bar*nr.net_weight).*(vec(xAB) .+ BBu)
        virt_bar .*= inv_M

        nr_bar = Tangent{NetworkL1Regularizer}(net_virtual=virt_bar)
        
        x_bar = similar(x)
        x_bar[:] .= nr.net_weight.*(vec(xAA) .+ ABu)
        x_bar .+= nr.l1_weight.*(sign.(x) .* nr.l1_feat_idx[1])
        x_bar .*= loss_bar
        x_bar .*= inv_M

        return nr_bar, x_bar
    end
    
    return loss, netreg_vec_pullback
end

###########################################
# Regularizer for Batch Arrays
###########################################

mutable struct BatchArrayReg 
    weight::Number
    counts::Tuple
    total::Number
end

@functor BatchArrayReg

Flux.trainable(bar::BatchArrayReg) = ()

function BatchArrayReg(ba::BatchArray; weight=1.0)
    row_counts = map(mat->vec(sum(mat,dims=1)), ba.row_batches)
    col_counts = map(length, ba.col_ranges)
    counts = row_counts .* col_counts
    total = sum(map(sum, counts))
    return BatchArrayReg(weight, counts, total)
end


function (reg::BatchArrayReg)(ba::BatchArray)
    return reg.weight*0.5*sum(map((c,v)->sum(c .* v .* v), reg.counts, ba.values)) / reg.total
end


function ChainRulesCore.rrule(reg::BatchArrayReg, ba::BatchArray)

    result = reg(ba)

    function batcharray_reg_pullback(result_bar)
        factor = result_bar*reg.weight
        val_bar = map((c,v) -> c.*v, reg.counts, ba.values) .* factor ./ reg.total
        return ChainRulesCore.NoTangent(),
               ChainRulesCore.Tangent{BatchArray}(values=val_bar)

    end

    return result, batcharray_reg_pullback
end


###########################################
# Now define a combined regularizer functor 
###########################################

mutable struct PMLayerReg
    cscale_reg::ClusterRegularizer
    cshift_reg::ClusterRegularizer
    bscale_reg::BatchArrayReg
    bshift_reg::BatchArrayReg
end

@functor PMLayerReg

function (reg::PMLayerReg)(layers::PMLayers)
    return 0.25*(reg.cscale_reg(layers.cscale.logsigma) 
                + reg.cshift_reg(layers.cshift.mu)
                + reg.bscale_reg(layers.bscale.logdelta)
                + reg.bshift_reg(layers.bshift.theta))
end


function ChainRulesCore.rrule(reg::PMLayerReg, layers::PMLayers)

    cscale_loss, 
    cscale_pullback = ChainRulesCore.rrule(reg.cscale_reg, layers.cscale.logsigma)
    
    cshift_loss, 
    cshift_pullback = ChainRulesCore.rrule(reg.cshift_reg, layers.cshift.mu)
    
    bscale_loss, 
    bscale_pullback = ChainRulesCore.rrule(reg.bscale_reg, layers.bscale.logdelta)
    
    bshift_loss, 
    bshift_pullback = ChainRulesCore.rrule(reg.bshift_reg, layers.bshift.theta)
   
    function pmlayer_reg_pullback(loss_bar)
        cscale_reg_bar, logsigma_bar = cscale_pullback(loss_bar)
        cshift_reg_bar, mu_bar = cshift_pullback(loss_bar)

        bscale_reg_bar, logdelta_bar = bscale_pullback(loss_bar)
        bshift_reg_bar, theta_bar = bshift_pullback(loss_bar)
     
        return ChainRulesCore.NoTangent(),
               #ChainRulesCore.Tangent{PMLayerReg}(cscale_reg=map(v->0.25.*v, cscale_reg_bar),
               #                                   cshift_reg=map(v->0.25.*v,cshift_reg_bar)),
               ChainRulesCore.Tangent{PMLayers}(cshift=ChainRules.Tangent{ColShift}(mu=map(v->0.25.*v, mu_bar)),
                                                cscale=ChainRules.Tangent{ColScale}(logsigma=map(v->0.25.*v, logsigma_bar)),
                                                bshift=ChainRules.Tangent{BatchShift}(theta=map(v->0.25.*v, theta_bar)),
                                                bscale=ChainRules.Tangent{BatchScale}(logdelta=map(v->0.25.*v, logdelta_bar)))
    end

    result = 0.25*(cscale_loss + cshift_loss + bscale_loss + bshift_loss)

    return result, pmlayer_reg_pullback
end



