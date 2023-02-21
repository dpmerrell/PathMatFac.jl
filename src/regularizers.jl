
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

    x_virtual::NTuple{K, <:AbstractVector}

    weight::Number
end

@functor NetworkRegularizer

function NetworkRegularizer(feature_ids, edgelists; epsilon=0.1, weight=1.0)

    N = length(feature_ids)
    K = length(edgelists)

    AA = Vector{SparseMatrixCSC}(undef, K) 
    AB = Vector{SparseMatrixCSC}(undef, K)
    BB = Vector{SparseMatrixCSC}(undef, K)
    x_virtual = Vector{Vector{Float64}}(undef, K)

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

        # Initialize values of virtual features
        x_virtual[k] = zeros(size(BB[k],1))

    end

    return NetworkRegularizer(Tuple(AA), Tuple(AB), 
                                         Tuple(BB), 
                                         Tuple(x_virtual),
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
        nr.x_virtual[k] .= - cg(nr.BB[k], xAB, nr.x_virtual[k])[1]

        net_loss = 0.0
        net_loss += quadratic(nr.AA[k], X[k,:])
        net_loss += 2*dot(xAB, nr.x_virtual[k])
        net_loss += quadratic(nr.BB[k], nr.x_virtual[k])
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
        nr.x_virtual[k] .= - cg(nr.BB[k], xAB, nr.x_virtual[k])[1]
        ABu[:,k] .= nr.AB[k]*nr.x_virtual[k]
        BBu = nr.BB[k]*nr.x_virtual[k]

        # Full loss computation
        net_loss = 0.0
        net_loss += 0.5*dot(xAA[k,:], X[k,:])
        net_loss += dot(X[k,:], ABu[:,k])
        net_loss += 0.5*dot(nr.x_virtual[k], BBu)
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
    nr.x_virtual[1] .= - cg(nr.BB[k], xAB, nr.x_virtual[1])[1]
    loss = 0.0
    loss += quadratic(nr.AA[1], x)
    loss += 2*dot(xAB, nr.x_virtual[k])
    loss += quadratic(nr.BB[1], nr.x_virtual[k])
    loss *= (0.5*nr.weight)

    return loss
end


function ChainRules.rrule(nr::NetworkRegularizer, x::AbstractVector)
    
    loss = 0.0

    xAA = transpose(x)*nr.AA[1]
    xAB = transpose(x)*nr.AB[1]
    nr.x_virtual[1] .= - cg(nr.BB[k], xAB, nr.x_virtual[1])[1]
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

@functor L1Regularizer

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
    group_labels::AbstractVector
    group_idx::AbstractMatrix{Float32}
    group_sizes::AbstractVector{Float32}
    biases::AbstractMatrix{Float32}
    bias_sizes::AbstractVector{Float32}
    weight::Number
end

@functor GroupRegularizer

function GroupRegularizer(group_labels::AbstractVector; weight=1.0, K=1)
    unq_labels = unique(group_labels)
    idx = ids_to_ind_mat(group_labels)
    sizes = vec(sum(idx, dims=1))
    
    n_groups = size(idx, 2)
    biases = zeros(Float32, K, n_groups)
    bias_sizes = zeros(Int32, n_groups)
 
    idx = sparse(idx)
    return GroupRegularizer(unq_labels, idx, sizes, biases, bias_sizes, weight)
end


function construct_new_group_reg(new_group_labels::AbstractVector,
                                 old_regularizer::GroupRegularizer, 
                                 old_array::AbstractMatrix)
    
    K = size(old_array,1)
    weight = old_regularizer.weight

    # Construct the new regularizer
    new_regularizer = GroupRegularizer(new_group_labels; weight=weight, K=K)
    
    # Collect the biases and bias sizes for the new regularizer,
    # whenever the new groups intersect the old groups
    old_intersect_idx, new_intersect_idx = keymatch(old_regularizer.group_labels,
                                                    new_regularizer.group_labels)

    old_means = (old_array * old_regularizer.group_idx) ./ transpose(old_regularizer.group_sizes) 
    old_intersect_means = old_means[:,old_intersect_idx]

    new_regularizer.biases[:,new_intersect_idx] .= old_intersect_means
    new_regularizer.bias_sizes[new_intersect_idx] .= old_regularizer.group_sizes[old_intersect_idx]

    return new_regularizer
end



#####################################
# regularize a vector
function (cr::GroupRegularizer)(x::AbstractVector)

    means = (transpose(cr.biases) .+ transpose(cr.group_idx)*x)./(cr.bias_sizes .+ cr.group_sizes)
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
   
    means = (transpose(cr.biases) .+ transpose(cr.group_idx)*x)./(cr.bias_sizes .+ cr.group_sizes)
    mean_vec = cr.group_idx * means
    diffs = x .- mean_vec

    function group_reg_pullback(loss_bar)
        return ChainRulesCore.NoTangent(), 
               Float32(cr.weight) .* diffs 
    end
    loss = 0.5*cr.weight*sum(diffs.*diffs)

    return loss, group_reg_pullback
end


####################################
# Regularize rows of a matrix
function (cr::GroupRegularizer)(X::AbstractMatrix)

    means = (transpose(cr.biases) .+ transpose(cr.group_idx)*transpose(X))./(cr.bias_sizes .+ cr.group_sizes)

    mean_rows = transpose(cr.group_idx * means)
    w_vec = cr.group_idx * (Float32(cr.weight) .+ cr.bias_sizes)
    diffs = X .- mean_rows

    return 0.5*sum(transpose(w_vec) .* diffs.*diffs) 
end


function ChainRulesCore.rrule(cr::GroupRegularizer, X::AbstractMatrix)

    means = (transpose(cr.biases) .+ transpose(cr.group_idx)*transpose(X))./(cr.bias_sizes .+ cr.group_sizes)
    mean_rows = transpose(cr.group_idx * means)
    w_vec = cr.group_idx * (Float32(cr.weight) .+ cr.bias_sizes)
    diffs = X .- mean_rows

    function group_reg_pullback(loss_bar)
        return ChainRulesCore.NoTangent(), 
               loss_bar .* (transpose(w_vec) .* diffs)
    end
    loss = 0.5*sum(transpose(w_vec) .* diffs.*diffs)

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

@functor CompositeRegularizer

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

function construct_X_reg(K, sample_ids, sample_conditions, sample_graphs, 
                         lambda_X_l2, lambda_X_condition, lambda_X_graph)
    regularizers = Any[x->0.0, x->0.0, x->0.0]
    if lambda_X_l2 != nothing
        regularizers[1] = x->(lambda_X_l2*0.5)*sum(x.*x)
    end

    if sample_conditions != nothing
        regularizers[2] = GroupRegularizer(sample_conditions; weight=lambda_X_condition, K=K)
    end

    if sample_graphs != nothing
        regularizers[3] = NetworkRegularizer(sample_ids, sample_graphs; weight=lambda_X_graph)
    end

    return construct_composite_reg(regularizers) 
end


###########################################
# Construct regularizer for Y matrix
###########################################

function construct_Y_reg(feature_ids, feature_graphs,
                         lambda_Y_l1, lambda_Y_selective_l1, lambda_Y_graph)

    regularizers = Any[x->0.0, x->0.0, x->0.0]
    if lambda_Y_l1 != nothing
        regularizers[1] =  y->(lambda_Y_l1*sum(abs.(y)))
    end

    if (feature_ids != nothing) & (feature_graphs != nothing)
        if lambda_Y_selective_l1 != nothing
            regularizers[2] = L1Regularizer(feature_ids, feature_graphs; 
                                            weight=lambda_Y_selective_l1)
        end
        if lambda_Y_graph != nothing
            regularizers[3] = NetworkRegularizer(feature_ids, feature_graphs;
                                                 weight=lambda_Y_graph)
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
    biases::Tuple
    bias_counts::Tuple
end

@functor BatchArrayReg 

function BatchArrayReg(ba::BatchArray; weight=1.0)
    row_counts = map(mat->vec(sum(mat,dims=1)), ba.row_batches)
    col_counts = map(length, ba.col_ranges)
    counts = row_counts .* col_counts
    biases = map(zero, ba.values)
    bias_counts = map(zero, counts)
    return BatchArrayReg(weight, counts, biases, bias_counts) 
end


function BatchArrayReg(feature_views::AbstractVector, batch_dict; weight=1.0)
    row_batches = [batch_dict[uv] for uv in unique(feature_views)] 
    values = [Dict(urb => 0.0 for urb in unique(rbv)) for rbv in row_batches]
    ba = BatchArray(feature_views, batch_dict, values)
    return BatchArrayReg(ba; weight=weight)
end


# Construct a new BatchArrayReg for the `new_batch_array`;
# but let its biases be informed by an `old_batch_array`.
# And mutate `new_batch_array`'s values to agree with those biases.
function construct_new_batch_reg!(new_batch_array, 
                                  old_batch_array_reg::BatchArrayReg,
                                  old_batch_array)
   
    new_reg = BatchArrayReg(new_batch_array; weight=old_batch_array_reg.weight)

    # Select the column ranges shared between the old and the new    
    old_col_intersect, new_col_intersect = keymatch(old_batch_array.col_range_ids,
                                                    new_batch_array.col_range_ids)

    # For each shared column range, set the new regularizer's
    # values, biases, and bias counts. 
    for (i_old, i_new) in zip(old_value_intersect, new_value_intersect)
        old_batch_intersect, new_batch_intersect = keymatch(old_batch_array.row_batch_ids[i_old],
                                                            new_batch_array.row_batch_ids[i_new])
        new_reg.biases[i_new][new_batch_intersect] .= old_batch_array.values[i_old][old_batch_intersect]
        new_reg.bias_counts[i_new][new_batch_intersect] .= old_batch_array_reg.counts[i_old][old_batch_intersect]
        new_batch_array.values[i_new][new_batch_intersect] .= old_batch_array.values[i_old][old_batch_intersect]
    end

    return new_reg 
end


function (reg::BatchArrayReg)(ba::BatchArray)
    diffs = ba.values .- reg.biases
    weights = map(v -> reg.weight .+ v, reg.bias_counts)
    return 0.5*sum(map((w,d)->sum(w .* d .* d), weights, diffs))
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

    weights = map(v -> reg.weight .+ v, reg.bias_counts)
    diffs = ba.values .- reg.biases
    wd = 0.5.* map((w,d)->w.*d, weights, diffs)

    function batcharray_reg_pullback(loss_bar)
        val_bar = loss_bar .* wd
        return ChainRulesCore.NoTangent(),
               ChainRulesCore.Tangent{BatchArray}(values=val_bar)

    end
    lss = sum(map((w,d) -> sum(w .* d), wd, diffs))

    return lss, batcharray_reg_pullback
end


############################################
# Construct regularizer for layers
############################################

mutable struct SequenceReg
    regs::Tuple
end

@functor SequenceReg

function (reg::SequenceReg)(seq)
    return sum(map((f,x)->f(x), reg.regs, seq.layers))
end


function construct_layer_reg(feature_views, batch_dict, lambda_layer)

    # Start with the regularizers for logsigma and mu
    # (the column parameters)
    regs = Any[x->0.0, x->0.0, x->0.0, x->0.0]
 
    # If a batch_dict is provided, add regularizers for
    # logdelta and theta (the batch parameters)
    if batch_dict != nothing
        regs[3] = BatchArrayReg(feature_views, batch_dict; weight=lambda_layer)
        regs[4] = BatchArrayReg(feature_views, batch_dict; weight=lambda_layer)
    end

    return SequenceReg(Tuple(regs))    
end




