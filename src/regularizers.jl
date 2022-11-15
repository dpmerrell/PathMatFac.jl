
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

    net_virtual::NTuple{K, <:AbstractVector} # Tuple of K vectors 
                                             # containing estimates of 
                                             # unobserved quantities
end


@functor NetworkRegularizer
Flux.trainable(nr::NetworkRegularizer) = (net_virtual=nr.net_virtual, )


function NetworkRegularizer(data_features, edgelists; epsilon=0.0)

    N = length(data_features)
    K = length(edgelists)

    AA = Vector{SparseMatrixCSC}(undef, K) 
    AB = Vector{SparseMatrixCSC}(undef, K)
    BB = Vector{SparseMatrixCSC}(undef, K)
    virtual = Vector{Float64}[]

    # For each of the networks
    for k=1:K

        ####################################
        # Construct network regularization
        edgelist = edgelists[k]

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

    end

    return NetworkRegularizer(Tuple(AA), Tuple(AB), 
                                         Tuple(BB), Tuple(virtual))
end

quadratic(u::AbstractVector, 
          X::AbstractMatrix, 
          v::AbstractVector) = dot(u, (X*v))

quadratic(X::AbstractMatrix, v::AbstractVector) = dot(v, (X*v))

##################################################
# Matrix row-regularization
##################################################

function (nr::NetworkRegularizer)(X::AbstractMatrix)

    loss = 0.0
    K,M = size(X)
    for k=1:K
        # Network-regularization
        net_loss = 0.0
        net_loss += quadratic(nr.AA[k], X[k,:])
        net_loss += 2*quadratic(X[k,:], nr.AB[k], nr.net_virtual[k])
        net_loss += quadratic(nr.BB[k], nr.net_virtual[k])
        net_loss *= 0.5
        loss += net_loss
    end
    loss /= (K*M)
    return loss
end


function ChainRules.rrule(nr::NetworkRegularizer, X::AbstractMatrix)

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
        loss += net_loss 
    end
    loss *= inv_KM

    function netreg_mat_pullback(loss_bar)

        # Gradient w.r.t. the network's virtual nodes
        virt_bar = map(similar, nr.net_virtual)
        for k=1:K
            virt_bar[k] .= (loss_bar*inv_KM).*(xAB[k] .+ BBu[k])
        end
        nr_bar = Tangent{NetworkRegularizer}(net_virtual=virt_bar)

        # Network regularization for the observed nodes
        X_bar = (xAA .+ transpose(ABu))
        X_bar .*= loss_bar
        X_bar .*= inv_KM

        return nr_bar, X_bar 
    end

    return loss, netreg_mat_pullback
end


#################################################
# Vector regularizer
#################################################

function (nr::NetworkRegularizer)(x::AbstractVector)

    loss = 0.0
    loss += quadratic(nr.AA[1], x)
    loss += 2*quadratic(x, nr.AB[1], nr.net_virtual[1])
    loss += quadratic(nr.BB[1], nr.net_virtual[1])
    loss *= 0.5

    return loss
end


function ChainRules.rrule(nr::NetworkRegularizer, x::AbstractVector)
    
    loss = 0.0
    inv_M = length(x)

    xAA = transpose(x)*nr.AA[1]
    xAB = transpose(x)*nr.AB[1]
    ABu = nr.AB[1]*nr.net_virtual[1]
    BBu = nr.BB[1]*nr.net_virtual[1]

    loss = 0.5*(dot(xAA, x) + 2*dot(x, ABu) + dot(nr.net_virtual[1], BBu))

    function netreg_vec_pullback(loss_bar)

        virt_bar = map(zero, nr.net_virtual)
        virt_bar[1] .= loss_bar.*(vec(xAB) .+ BBu)
        virt_bar .*= inv_M

        nr_bar = Tangent{NetworkRegularizer}(net_virtual=virt_bar)
        
        x_bar = similar(x)
        x_bar[:] .= vec(xAA) .+ ABu
        x_bar .*= loss_bar
        x_bar .*= inv_M

        return nr_bar, x_bar
    end
    
    return loss, netreg_vec_pullback
end


##########################################################
# L1 regularizer
##########################################################

mutable struct L1Regularizer
    l1_idx::AbstractMatrix{Bool} # Boolean matrix indicates which entries
                                 # are L1-regularized
end

@functor L1Regularizer
Flux.trainable(lr::L1Regularizer) = ()

function L1Regularizer(data_features::Vector, l1_features::Vector)

    N = length(data_features)
    K = length(l1_features)

    l1_idx = zeros(Bool, K, N)

    for (k, l1_feat) in enumerate(l1_features)
        l1_idx[k,:] .= map(x->in(x, l1_feat), data_features) 
    end

    return L1Regularizer(l1_idx)
end
                        

function (reg::L1Regularizer)(X::AbstractMatrix)
    K, N = size(X)
    return sum(abs.(reg.l1_idx .* X)) / (K*N)
end


##########################################################
# Network-L1 regularizer
##########################################################

mutable struct NetworkL1Regularizer{K}

    net_reg::NetworkRegularizer{K}
    net_weight::Number

    l1_reg::L1Regularizer
    l1_weight::Number
end

@functor NetworkL1Regularizer
Flux.trainable(nr::NetworkL1Regularizer) = (net_reg=nr.net_reg,)


function NetworkL1Regularizer(data_features::Vector, network_edgelists::Vector, l1_features::Vector;
                              net_weight=1.0, epsilon=1.0, l1_weight=1.0)

    l1_reg = L1Regularizer(data_features, l1_features)
    net_reg = NetworkRegularizer(data_features, network_edgelists; epsilon=epsilon)
    return NetworkL1Regularizer(net_reg, net_weight,
                                l1_reg, l1_weight)
end


##################################################
# Regularize the rows of a matrix

function (nr::NetworkL1Regularizer)(X::AbstractMatrix)
    return nr.net_weight*nr.net_reg(X) + nr.l1_weight*nr.l1_reg(X)
end


function ChainRulesCore.rrule(nr::NetworkL1Regularizer, X::AbstractMatrix)

    l1_lss, l1_pb = Zygote.pullback((r, x) -> r(x), nr.l1_reg, X)
    net_lss, net_pb = Zygote.pullback((r, x) -> r(x), nr.net_reg, X) 

    function networkl1_pullback(loss_bar)
        _, l1_X_bar = l1_pb(loss_bar)
        net_reg_bar, net_X_bar = net_pb(loss_bar)
        X_bar = nr.l1_weight.*l1_X_bar .+ nr.net_weight.*net_X_bar 
 
        return ChainRulesCore.Tangent{NetworkL1Regularizer}(net_reg=net_reg_bar),
               X_bar
    end

    return l1_lss*nr.l1_weight + net_lss*nr.net_weight, networkl1_pullback
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
               ChainRulesCore.Tangent{PMLayers}(cshift=ChainRules.Tangent{ColShift}(mu=map(v->0.25.*v, mu_bar)),
                                                cscale=ChainRules.Tangent{ColScale}(logsigma=map(v->0.25.*v, logsigma_bar)),
                                                bshift=ChainRules.Tangent{BatchShift}(theta=map(v->0.25.*v, theta_bar)),
                                                bscale=ChainRules.Tangent{BatchScale}(logdelta=map(v->0.25.*v, logdelta_bar)))
    end

    result = 0.25*(cscale_loss + cshift_loss + bscale_loss + bshift_loss)

    return result, pmlayer_reg_pullback
end



