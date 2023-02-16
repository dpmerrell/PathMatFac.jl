
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

    BB_chol::NTuple # Tuple of K cholesky factorizations 

    weight::Number
end


function NetworkRegularizer(feature_ids, edgelists; epsilon=0.1, weight=1.0)

    N = length(feature_ids)
    K = length(edgelists)

    AA = Vector{SparseMatrixCSC}(undef, K) 
    AB = Vector{SparseMatrixCSC}(undef, K)
    BB = Vector{SparseMatrixCSC}(undef, K)
    BB_chol = Vector{Any}(undef, K)

    # For each of the networks
    for k=1:K

        ####################################
        # Construct network regularization
        edgelist = edgelists[k]

        # Extract the nodes from this edgelist
        net_nodes = get_all_nodes(edgelist)

        # Determine which are virtual and which are observed
        net_virtual_nodes = setdiff(net_nodes, feature_ids)
        net_features = setdiff(net_nodes, net_virtual_nodes)
        
        # append the virtual features to data
        net_virtual_nodes = sort(collect(net_virtual_nodes))
        all_nodes = vcat(feature_ids, net_virtual_nodes)
        node_to_idx = value_to_idx(all_nodes) 

        # Construct a sparse matrix encoding this network
        spmat = edgelist_to_spmat(edgelist, node_to_idx; epsilon=epsilon)

        # Split this matrix into observed/unobserved blocks
        N_total = size(spmat, 2)
        AA[k] = csc_select(spmat, 1:N, 1:N)
        AB[k] = csc_select(spmat, 1:N, (N+1):N_total)
        BB[k] = csc_select(spmat, (N+1):N_total, (N+1):N_total)

        # Pre-factorize the BB blocks
        BB_chol[k] = cholesky(BB[k]) 

    end

    return NetworkRegularizer(Tuple(AA), Tuple(AB), 
                                         Tuple(BB), 
                                         Tuple(BB_chol),
                                         weight)
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
        xAB = transpose(transpose(X[k,:])*nr.AB[k])
        u = -(nr.BB_chol[k]\xAB)

        net_loss = 0.0
        net_loss += quadratic(nr.AA[k], X[k,:])
        net_loss += 2*dot(xAB,u)
        net_loss += quadratic(nr.BB[k], u)
        net_loss *= 0.5
        loss += net_loss
    end
    loss *= nr.weight
    return loss
end


function ChainRules.rrule(nr::NetworkRegularizer, X::AbstractMatrix)

    K, MA = size(X)
    MB = size(nr.BB, 1)

    # Pre-allocate these matrix-vector products
    xAA = similar(X, K,MA)
    ABu = similar(X, MA,K)

    loss = 0.0
    for k=1:K
        # Network-regularization
        # Parts of the gradients
        xAA[k,:] .= vec(transpose(X[k,:])*nr.AA[k])
        xAB = vec(transpose(X[k,:])*nr.AB[k])
        u = -(nr.BB_chol[k]\xAB)
        ABu[:,k] .= nr.AB[k]*u
        BBu = nr.BB[k]*u

        # Full loss computation
        net_loss = 0.0
        net_loss += 0.5*dot(xAA[k,:], X[k,:])
        net_loss += dot(X[k,:], ABu[:,k])
        net_loss += 0.5*dot(u, BBu)
        loss += net_loss 
    end
    loss *= nr.weight

    function netreg_mat_pullback(loss_bar)

        # Network regularization for the observed nodes
        X_bar = (xAA .+ transpose(ABu))
        X_bar .*= loss_bar
        X_bar .*= nr.weight

        return NoTangent(), X_bar 
    end

    return loss, netreg_mat_pullback
end


#################################################
# Vector regularizer
#################################################

function (nr::NetworkRegularizer)(x::AbstractVector)

    xAB = vec(transpose(x)*nr.AB[1])
    u = -(nr.BB_chol[1]\xAB)
    loss = 0.0
    loss += quadratic(nr.AA[1], x)
    loss += 2*dot(xAB, u)
    loss += quadratic(nr.BB[1], u)
    loss *= (0.5*nr.weight)

    return loss
end


function ChainRules.rrule(nr::NetworkRegularizer, x::AbstractVector)
    
    loss = 0.0

    xAA = transpose(x)*nr.AA[1]
    xAB = transpose(x)*nr.AB[1]
    u = -(nr.BB_chol[1]\xAB)
    ABu = nr.AB[1]*u
    BBu = nr.BB[1]*u

    loss = 0.5*(dot(xAA, x) + 2*dot(x, ABu) + dot(u, BBu))

    function netreg_vec_pullback(loss_bar)

        x_bar = similar(x)
        x_bar[:] .= vec(xAA) .+ ABu
        x_bar .*= loss_bar
        x_bar .*= nr.weight 

        return NoTangent(), x_bar
    end
    
    return loss, netreg_vec_pullback
end


##########################################################
# Selective L1 regularizer
##########################################################

mutable struct L1Regularizer
    l1_idx::AbstractMatrix{Bool} # Boolean matrix indicates which entries
                                 # are L1-regularized
    weight::Number
end

function L1Regularizer(feature_ids::Vector, edgelists::Vector; weight=1.0)

    l1_features = compute_nongraph_nodes(feature_ids, edgelists) 
    N = length(feature_ids)
    K = length(l1_features)

    l1_idx = zeros(Bool, K, N)

    for (k, l1_feat) in enumerate(l1_features)
        l1_idx[k,:] .= map(x->in(x, l1_feat), feature_ids) 
    end

    return L1Regularizer(l1_idx, weight)
end
                        
function (reg::L1Regularizer)(X::AbstractArray)
    return reg.weight*sum(abs.(reg.l1_idx .* X)) 
end


function ChainRulesCore.rrule(reg::L1Regularizer, X::AbstractArray)

    sel_X = reg.l1_idx .* X
    lss = reg.weight*sum(abs.(sel_X))

    function L1Regularizer_pullback(loss_bar)
        X_bar = (loss_bar*reg.weight).*sign.(sel_X)
        return ChainRulesCore.NoTangent(), X_bar
    end    

    return lss, L1Regularizer_pullback
end


##########################################################
# Group Regularizer
##########################################################

mutable struct GroupRegularizer 
    group_idx::AbstractMatrix{Float32}
    group_sizes::AbstractVector{Int32}
    weight::Number
end


function GroupRegularizer(group_labels::AbstractVector; weight=1.0)
    idx = ids_to_ind_mat(group_labels)
    sizes = vec(sum(idx, dims=1))
    idx = sparse(idx)
    return GroupRegularizer(idx, sizes, weight)
end


#####################################
# regularize a vector
function (cr::GroupRegularizer)(x::AbstractVector)

    means = (transpose(cr.group_idx) * x) ./ cr.group_sizes
    mean_vec = cr.group_idx * means
    diffs = x .- mean_vec
    return 0.5*cr.weight*sum(diffs.*diffs)
end

function (cr::GroupRegularizer)(layer::ColScale)
    return cr(layer.logsigma)
end

function (cr::GroupRegularizer)(layer::ColShift)
    return cr(layer.mu)
end

function (cr::GroupRegularizer)(layer::FrozenLayer)
    return 0.0
end


function ChainRulesCore.rrule(cr::GroupRegularizer, x::AbstractVector)
   
    means = (transpose(cr.group_idx) * x) ./ cr.group_sizes
    mean_vec = cr.group_idx * means
    diffs = x .- mean_vec

    function group_reg_pullback(loss_bar)
        return ChainRulesCore.NoTangent(), 
               (cr.weight*loss_bar) .* diffs 
    end
    loss = 0.5*cr.weight*sum(diffs.*diffs)

    return loss, group_reg_pullback
end


####################################
# Regularize rows of a matrix
function (cr::GroupRegularizer)(X::AbstractMatrix)

    means = (transpose(cr.group_idx) * transpose(X)) ./ cr.group_sizes
    mean_rows = transpose(cr.group_idx * means)
    diffs = X .- mean_rows

    return (0.5*cr.weight)*sum(diffs.*diffs) 
end


function ChainRulesCore.rrule(cr::GroupRegularizer, X::AbstractMatrix)

    means = (transpose(cr.group_idx) * transpose(X)) ./ cr.group_sizes
    mean_rows = transpose(cr.group_idx * means)
    diffs = X .- mean_rows

    function group_reg_pullback(loss_bar)
        return ChainRulesCore.NoTangent(), 
               (cr.weight*loss_bar) .* diffs
    end
    loss = 0.5*cr.weight*sum(diffs.*diffs)

    return loss, group_reg_pullback
end


###########################################
# Function for constructing "composite" 
# regularizers. I.e., combinations of
# regularizers that all act on the same parameter.
###########################################

mutable struct CompositeRegularizer
    regularizers::Tuple
end

function construct_composite_reg(regs::AbstractVector)
    if length(regs) == 0
        regs = [ x->0.0 ]
    end
    return CompositeRegularizer(Tuple(regs))
end

function (cr::CompositeRegularizer)(x)
    return sum(map(f->f(x), cr.regularizers))
end


###########################################
# Construct regularizer for X matrix
###########################################

function construct_X_reg(sample_ids, sample_conditions, sample_graphs, 
                         lambda_X_l2, lambda_X_condition, lambda_X_graph)
    regularizers = []
    if lambda_X_l2 != nothing
        push!(regularizers, x->(lambda_X_l2*0.5)*sum(x.*x))
    end

    if sample_conditions != nothing
        push!(regularizers, GroupRegularizer(sample_conditions; weight=lambda_X_condition))
    end

    if sample_graphs != nothing
        push!(regularizers, NetworkRegularizer(sample_ids, sample_graphs; weight=lambda_X_graph))
    end

    return construct_composite_reg(regularizers) 
end


###########################################
# Construct regularizer for Y matrix
###########################################

function construct_Y_reg(feature_ids, feature_graphs,
                         lambda_Y_l1, lambda_Y_selective_l1, lambda_Y_graph)

    regularizers = []
    if lambda_Y_l1 != nothing
        push!(regularizers, y->(lambda_Y_l1*sum(abs.(y))))
    end

    if (feature_ids != nothing) & (feature_graphs != nothing)
        if lambda_Y_selective_l1 != nothing
            push!(regularizers, L1Regularizer(feature_ids, feature_graphs; 
                                              weight=lambda_Y_selective_l1))
        end
        if lambda_Y_graph != nothing
            push!(regularizers, NetworkRegularizer(feature_ids, feature_graphs;
                                                   weight=lambda_Y_graph))
        end
    end

    return construct_composite_reg(regularizers) 
end


###########################################
# Regularizer for Batch Arrays
###########################################

mutable struct BatchArrayReg 
    weight::Number
    counts::Tuple
end

function BatchArrayReg(ba::BatchArray; weight=1.0)
    row_counts = map(mat->vec(sum(mat,dims=1)), ba.row_batches)
    col_counts = map(length, ba.col_ranges)
    counts = row_counts .* col_counts
    return BatchArrayReg(weight, counts) 
end


function BatchArrayReg(feature_views::AbstractVector, batch_dict; weight=1.0)
    row_batches = [batch_dict[uv] for uv in unique(feature_views)] 
    values = [Dict(urb => 0.0 for urb in unique(rbv)) for rbv in row_batches]
    ba = BatchArray(feature_views, row_batches, values)
    return BatchArrayReg(ba; weight=weight)
end


function (reg::BatchArrayReg)(ba::BatchArray)
    return reg.weight*0.5*sum(map((c,v)->sum(c .* v .* v), reg.counts, ba.values))
end

function (reg::BatchArrayReg)(layer::BatchScale)
    return reg(layer.logdelta)
end

function (reg::BatchArrayReg)(layer::BatchShift)
    return reg(layer.theta)
end

function (cr::BatchArrayReg)(layer::FrozenLayer)
    return 0.0
end

function ChainRulesCore.rrule(reg::BatchArrayReg, ba::BatchArray)

    result = reg(ba)

    function batcharray_reg_pullback(result_bar)
        factor = result_bar*reg.weight
        val_bar = map((c,v) -> c.*v, reg.counts, ba.values) .* factor 
        return ChainRulesCore.NoTangent(),
               ChainRulesCore.Tangent{BatchArray}(values=val_bar)

    end

    return result, batcharray_reg_pullback
end


############################################
# Construct regularizer for layers
############################################

mutable struct SequenceReg
    regs::Tuple
end


function (reg::SequenceReg)(seq)
    return sum(map((f,x)->f(x), reg.regs, seq.layers))
end


function construct_layer_reg(feature_views, batch_dict, lambda_layer)

    # Start with the regularizers for logsigma and mu
    # (the column parameters)
    regs = Any[x->0.0, x->0.0]
 
    # If a batch_dict is provided, add regularizers for
    # logdelta and theta (the batch parameters)
    if batch_dict != nothing
        append!(regs,Any[BatchArrayReg(feature_views, batch_dict; weight=lambda_layer),
                         BatchArrayReg(feature_views, batch_dict; weight=lambda_layer)])
    end

    return SequenceReg(Tuple(regs))    
end




