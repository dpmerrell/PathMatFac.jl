
import Base: ==


##########################################################
# L2 regularizer
##########################################################

mutable struct L2Regularizer
    weights::AbstractVector
end

@functor L2Regularizer

function L2Regularizer(K::Integer, w::Number)
    return L2Regularizer(fill(w, K)) 
end

function (reg::L2Regularizer)(X::AbstractMatrix)
    return 0.5*sum(reg.weights .* sum(X .* X, dims=2))
end

function ChainRulesCore.rrule(reg::L2Regularizer, X::AbstractMatrix)
    g = reg.weights .* X

    function L2Regularizer_pullback(loss_bar)
        return NoTangent(), loss_bar .* g
    end

    return 0.5*sum(g .* X), L2Regularizer_pullback
end

function (reg::L2Regularizer)(x::AbstractVector)
    return reg(transpose(x))
end

function reweight_eb!(reg::L2Regularizer, X::AbstractMatrix; mixture_p=1.0)
    row_vars = vec(var(X, dims=2))
    row_precs = 1 ./ row_vars
    reg.weights .= (mixture_p .* row_precs)
end 

function reweight_eb!(reg::L2Regularizer, x::AbstractVector; mixture_p=1.0)
    reweight_eb!(reg, transpose(x); mixture_p=mixture_p)
end 


##########################################################
# L1 regularizer
##########################################################

mutable struct L1Regularizer
    weights::AbstractVector
end

@functor L1Regularizer

function L1Regularizer(K::Integer, w::Number)
    return L1Regularizer(fill(w, K)) 
end

function (reg::L1Regularizer)(X::AbstractMatrix)
    return sum(reg.weights .* sum(abs.(X), dims=2))
end

function ChainRulesCore.rrule(reg::L1Regularizer, X::AbstractMatrix)

    function L1Regularizer_pullback(loss_bar)
        return NoTangent(), (loss_bar .* reg.weights) .* sign.(X)
    end

    return sum(reg.weights .* sum(abs.(X), dims=2)), L1Regularizer_pullback
end

function (reg::L1Regularizer)(x::AbstractVector)
    return reg(transpose(x))
end

function reweight_eb!(reg::L1Regularizer, X::AbstractMatrix; mixture_p=1.0)
    row_vars = vec(var(X, dims=2))
    row_precs = 1 ./ row_vars
    reg.weights .= (mixture_p .* row_precs)
end 

function reweight_eb!(reg::L1Regularizer, x::AbstractVector; mixture_p=1.0)
    reweight_eb!(reg, transpose(x); mixture_p=mixture_p)
end 


##########################################################
# Selective L1 regularizer
##########################################################

mutable struct SelectiveL1Reg
    l1_idx::AbstractMatrix{Bool} # Boolean matrix indicates which entries
                                 # are L1-regularized
    weight::AbstractVector 
end

@functor SelectiveL1Reg

function SelectiveL1Reg(feature_ids::Vector, edgelists::Vector; weight=1.0)

    l1_features = compute_nongraph_nodes(feature_ids, edgelists) 
    N = length(feature_ids)
    K = length(l1_features)

    l1_idx = zeros(Bool, K, N)

    for (k, l1_feat) in enumerate(l1_features)
        l1_idx[k,:] .= map(x->in(x, l1_feat), feature_ids) 
    end

    return SelectiveL1Reg(l1_idx, fill(weight, K))
end
                        

function (reg::SelectiveL1Reg)(X::AbstractArray)
    return sum(reg.weight .* sum(abs.(reg.l1_idx .* X), dims=2)) 
end


function ChainRulesCore.rrule(reg::SelectiveL1Reg, X::AbstractArray)

    sel_X = reg.l1_idx .* X
    lss = sum(reg.weight .* sum(abs.(sel_X), dims=2))

    function SelectiveL1Reg_pullback(loss_bar)
        X_bar = (loss_bar.*reg.weight) .* sign.(sel_X)
        return ChainRulesCore.NoTangent(), X_bar
    end    

    return lss, SelectiveL1Reg_pullback
end


function reweight_eb!(reg::SelectiveL1Reg, X::AbstractMatrix; mixture_p=1.0)
    # Given an empirical variance V,
    # the corresponding Laplace scale b = sqrt(V/2) 
    sel_X = reg.l1_idx .* X
    mean_x = mean(sel_X, dims=2)
    mean_x2 = mean(sel_X.*sel_X, dims=2)
    var_x = vec(mean_x2 .- (mean_x.*mean_x))
    new_weights = mixture_p .* sqrt.(2 ./ var_x)
    new_weights[(!isfinite).(new_weights)] .= 1
    reg.weight .= new_weights
end


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

    cur_weights::AbstractVector
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
        scale_spmat!(spmat, weight)

        #TODO: "REWEIGHT" THESE MATRICES S.T. THEY TEND TO
        #      GENERATE VECTORS OF EQUAL MAGNITUDE
        #TODO: THINK ABOUT HOW TO CHOOSE COMPARATIVE WEIGHT
        #      OF NETWORK AND DIAGONAL ENTRIES

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
                                         fill(weight, K))
end

quadratic_form(u::AbstractVector, 
               X::AbstractMatrix, 
               v::AbstractVector) = dot(u, (X*v))

quadratic_form(X::AbstractMatrix, v::AbstractVector) = dot(v, (X*v))


function (nr::NetworkRegularizer)(X::AbstractMatrix)

    loss = 0.0
    K,M = size(X)
    for k=1:K
        # Network-regularization
        xAB = transpose(transpose(X[k,:])*nr.AB[k])
        nr.x_virtual[k] .= - cg(nr.BB[k], xAB, nr.x_virtual[k])[1]

        net_loss = 0.0
        net_loss += quadratic_form(nr.AA[k], X[k,:])
        net_loss += 2*dot(xAB, nr.x_virtual[k])
        net_loss += quadratic_form(nr.BB[k], nr.x_virtual[k])
        net_loss *= 0.5
        loss += net_loss
    end
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

    function netreg_mat_pullback(loss_bar)

        # Network regularization for the observed nodes
        X_bar = (xAA .+ transpose(ABu))
        X_bar .*= loss_bar

        return NoTangent(), X_bar 
    end

    return loss, netreg_mat_pullback
end

function (nr::NetworkRegularizer)(x::AbstractVector)
    return nr(transpose(x))
end

#TODO revisit this after thinking about it some more
function reweight_eb!(nr::NetworkRegularizer, X::AbstractMatrix; mixture_p=1.0)
    
    # Compute weights from X
    K, N = size(X)
    row_precs = vec(mixture_p ./ var(X, dims=2))

    # Rescale the blocks of each sparse matrix
    new_over_old = cpu(row_precs ./ nr.cur_weights)
    for k=1:K
        scale_spmat!(nr.AA[k], new_over_old[k])
        scale_spmat!(nr.AB[k], new_over_old[k])
        scale_spmat!(nr.BB[k], new_over_old[k])
    end

    nr.cur_weights .= row_precs
end

 

##########################################################
# Group Regularizer
##########################################################

mutable struct GroupRegularizer
    group_labels::AbstractVector
    group_idx::AbstractMatrix{Float32}
    group_weights::AbstractVector
    group_sizes::AbstractVector
end

@functor GroupRegularizer

function GroupRegularizer(group_labels::AbstractVector; weight=1.0, K=1)
    unq_labels = unique(group_labels)
    idx = ids_to_ind_mat(group_labels)
    idx = sparse(idx)
    group_sizes = transpose(idx)*ones(length(group_labels))
    n_groups = size(idx, 2)
    weights = fill(weight, n_groups)
    return GroupRegularizer(unq_labels, idx, weights, group_sizes)
end


function construct_new_group_reg(new_group_labels::AbstractVector,
                                 old_regularizer::GroupRegularizer, 
                                 old_array::AbstractMatrix; 
                                 mixture_p=1.0)
    # Make a copy of the old regularizer and reweight it in an
    # empirical Bayes fashion.
    old_reg_copy = deepcopy(old_regularizer)
    reweight_eb!(old_reg_copy, old_array; mixture_p=mixture_p)

    # Figure out which groups are shared between old and new
    unq_new_labels = unique(new_group_labels)
    old_intersect_idx, new_intersect_idx = keymatch(old_reg_copy.group_labels,
                                                    unq_new_labels)

    # Construct a vector of group weights for the new regularizer.
    # By default, set it to the mean of the old weights 
    expected_weight = mean(old_reg_copy.group_weights)
    new_weights = fill(expected_weight, length(unq_new_labels))

    # Copy weights from the old regularizer to the new one,
    # whenever appropriate
    new_weights[new_intersect_idx] .= old_reg_copy.group_weights[old_intersect_idx]

    # Finally: construct the sparse index matrix for the new regularizer
    new_idx = ids_to_ind_mat(new_group_labels)
    new_idx = sparse(new_idx)
    new_group_sizes = transpose(new_idx) * ones(length(new_group_labels))

    # Return the completed regularizer
    return GroupRegularizer(unq_new_labels, new_idx, new_weights, new_group_sizes)
end

function reweight_eb!(gr::GroupRegularizer, X::AbstractMatrix; mixture_p=1.0)
    # Compute the group-wise variances from X
    means = transpose((X * gr.group_idx)) ./ gr.group_sizes
    mean_mat = transpose(gr.group_idx * means)
    diffs = X .- mean_mat
    diff_sq = diffs.*diffs
    group_vars = transpose((diff_sq * gr.group_idx)) ./ gr.group_sizes
    
    # Set the group weights to their inverse variances
    gr.group_weights .= mixture_p ./ vec(mean(group_vars, dims=2))
end


function (gr::GroupRegularizer)(X::AbstractMatrix)

    means = transpose((X * gr.group_idx)) ./ gr.group_sizes
    mean_mat = transpose(gr.group_idx * means)
    weight_vec = gr.group_idx * gr.group_weights
    diffs = X .- mean_mat

    return 0.5*sum(transpose(weight_vec) .* diffs.*diffs) 
end


function ChainRulesCore.rrule(gr::GroupRegularizer, X::AbstractMatrix)

    means = transpose((X * gr.group_idx)) ./ gr.group_sizes
    mean_mat = transpose(gr.group_idx * means)
    weight_vec = gr.group_idx * gr.group_weights
    diffs = X .- mean_mat
    g = transpose(weight_vec) .* diffs

    function group_reg_pullback(loss_bar)
        return ChainRulesCore.NoTangent(), 
               loss_bar.* (g .* diffs)
    end
    loss = 0.5*sum(g .* diffs)

    return loss, group_reg_pullback
end

# Define behaviors on various types

function (gr::GroupRegularizer)(x::AbstractVector)
    return gr(transpose(x))
end

function (gr::GroupRegularizer)(layer::ColScale)
    return gr(layer.logsigma)
end

function (gr::GroupRegularizer)(layer::ColShift)
    return gr(layer.mu)
end

function (gr::GroupRegularizer)(layer::FrozenLayer)
    return 0.0
end

function reweight_eb!(gr::GroupRegularizer, x::AbstractVector; mixture_p=1.0)
    reweight_eb!(gr, transpose(x); mixture_p=mixture_p)
end

function reweight_eb!(gr::GroupRegularizer, cs::ColShift; mixture_p=1.0)
    reweight_eb!(gr, cs.mu; mixture_p=mixture_p)
end

function reweight_eb!(gr::GroupRegularizer, cs::ColScale; mixture_p=1.0)
    reweight_eb!(gr, cs.logsigma; mixture_p=mixture_p)
end


###########################################
# Automatic Relevance Determination (ARD) 
###########################################

mutable struct ARDRegularizer
    alpha::Number
    beta::Number
    weight::Number
end

@functor ARDRegularizer

function ARDRegularizer(;alpha=Float32(1e-6), 
                         beta=Float32(1e-6),
                         weight=1.0)
    return ARDRegularizer(alpha, beta, weight)
end


function (ard::ARDRegularizer)(X::AbstractMatrix)
    b = 1 .+ (0.5/ard.beta).*(X.*X)
    return (0.5 + ard.alpha)*sum(log.(b))
end


function ChainRulesCore.rrule(ard::ARDRegularizer, X::AbstractMatrix)
    
    b = 1 .+ (0.5/ard.beta).*(X.*X)
    function ard_pullback(loss_bar)
        return NoTangent(), (loss_bar/ard.beta)*(0.5 + ard.alpha) .* X ./ b 
    end

    return (0.5 .+ ard.alpha)*sum(log.(b)), ard_pullback
end


#################################################
# Struct representing a mixture of regularizers 
#################################################

mutable struct CompositeRegularizer
    regularizers::Tuple
    mixture_p::AbstractVector
end

@functor CompositeRegularizer
Flux.trainable(cr::CompositeRegularizer) = (regularizers=cr.regularizers,)


function construct_composite_reg(regs::AbstractVector, mixture_p::AbstractVector)
    if length(regs) == 0
        regs = [ x->0.0 ]
        mixture_p = [0]
    end
    return CompositeRegularizer(Tuple(regs), mixture_p)
end


function reweight_eb!(cr::CompositeRegularizer, X; super_mixture_p=1.0)
    for (reg, p) in zip(cr.regularizers, cr.mixture_p)
        reweight_eb!(reg, X; mixture_p=p*super_mixture_p) 
    end
end


function (cr::CompositeRegularizer)(x)
    return sum(map((f,p)->p*f(x), cr.regularizers, cr.mixture_p))
end


######################################################
# Construct regularizer for X matrix
######################################################

function construct_X_reg(K, M, sample_ids, sample_conditions, sample_graphs, 
                         lambda_X_l2, lambda_X_condition, lambda_X_graph,
                         Y_ard, Y_geneset_ard)

    # If we're doing ARD on Y, then we require X 
    # to be quadratic or group regularized. 
    # Regularization weights will be set by empirical Bayes.
    if (Y_ard | Y_geneset_ard)
        if sample_conditions != nothing
            return GroupRegularizer(sample_conditions; weight=1.0, K=K)
        else
            return L2Regularizer(K, 1.0)
        end
    end

    regularizers = Any[x->0.0, x->0.0, x->0.0]
    mixture_p = zeros(3)
    if lambda_X_l2 != nothing
        regularizers[1] = L2Regularizer(K, lambda_X_l2)
        mixture_p[1] = 1
    end

    if sample_conditions != nothing
        regularizers[2] = GroupRegularizer(sample_conditions; weight=lambda_X_condition, K=K)
        mixture_p[2] = 1
    end

    if sample_graphs != nothing
        regularizers[3] = NetworkRegularizer(sample_ids, sample_graphs; weight=lambda_X_graph)
        mixture_p[3] = 1
    end

    mixture_p ./= sum(mixture_p)
    return construct_composite_reg(regularizers, mixture_p) 
end


#########################################################
# Construct regularizer for Y matrix
#########################################################

function construct_Y_reg(K, N, feature_ids, feature_sets, feature_graphs,
                         lambda_Y_l1, lambda_Y_selective_l1, lambda_Y_graph,
                         Y_ard, Y_geneset_ard)

    # If either of the ARD flags are set `true`, then
    # they take priority over the other regularizers.
    if Y_geneset_ard
        return construct_featureset_ard(K, feature_ids, feature_sets) 
    end
    if Y_ard
        return ARDRegularizer() 
    end

    # If neither ARD flag is set `true`, then 
    # construct a regularizer from any other flags.
    # (Default is *no* regularization) 
    regularizers = Any[x->0.0, x->0.0, x->0.0]
    mixture_p = zeros(3)
    if lambda_Y_l1 != nothing
        regularizers[1] =  L1Regularizer(K, lambda_Y_l1)
        mixture_p[1] = 1
    end

    if (feature_ids != nothing) & (feature_graphs != nothing)
        if lambda_Y_selective_l1 != nothing
            regularizers[2] = SelectiveL1Reg(feature_ids, feature_graphs; 
                                             weight=lambda_Y_selective_l1)
            mixture_p[2] = 1
        end
        if lambda_Y_graph != nothing
            regularizers[3] = NetworkRegularizer(feature_ids, feature_graphs;
                                                 weight=lambda_Y_graph)
            mixture_p[3] = 1
        end
    end

    mixture_p ./= sum(mixture_p)
    return construct_composite_reg(regularizers, mixture_p) 
end


##########################################################
# Regularizer for Batch Array structs
##########################################################

mutable struct BatchArrayReg 
    weights::Tuple
end

@functor BatchArrayReg 

function BatchArrayReg(ba::BatchArray; weight=1.0)
    N_col_batches = length(ba.col_ranges)
    return BatchArrayReg(Tuple(fill(weight, N_col_batches))) 
end


function BatchArrayReg(feature_views::AbstractVector; weight=1.0)
    unq_feature_views = unique(feature_views)
    return BatchArrayReg(Tuple(fill(weight, length(unq_feature_views))))
end


function (reg::BatchArrayReg)(ba::BatchArray)
    return 0.5*sum(map((w,v)->w*sum(v .* v), reg.weights, ba.values))
end


function ChainRulesCore.rrule(reg::BatchArrayReg, ba::BatchArray)

    grad = map((w,v)->w.*v, reg.weights, ba.values)

    function batcharray_reg_pullback(loss_bar)
        val_bar = loss_bar .* grad
        return ChainRulesCore.NoTangent(),
               ChainRulesCore.Tangent{BatchArray}(values=val_bar)

    end
    lss = 0.5*sum(map((g,v) -> sum(g .* v), grad, ba.values))

    return lss, batcharray_reg_pullback
end


function reweight_eb!(reg::BatchArrayReg, A::BatchArray; mixture_p=1.0)
    n_col_range = length(A.col_ranges)
    new_weights = zeros(n_col_range)
     
    for i=1:n_col_range
        M, n_batches = size(A.row_batches[i])
        batch_sizes = ones(1, M) * A.row_batches[i]
        all_values = A.row_batches[i] * A.values[i]
        batch_param_var = var(all_values)
        new_weights[i] = mixture_p ./ batch_param_var
    end

    reg.weights = Tuple(new_weights)
end


# Construct a new BatchArrayReg for the `new_batch_array`;
# but let its biases be informed by an `old_batch_array`.
# And mutate `new_batch_array`'s values to agree with those biases.
function construct_new_batch_reg!(new_batch_array::BatchArray, 
                                  old_batch_array_reg::BatchArrayReg,
                                  old_batch_array)
 
    # Make a copy of the old regularizer, and update its weights
    # in an empirical Bayes fashion. 
    old_reg_copy = deepcopy(old_batch_array)
    reweight_eb!(old_reg_copy, old_batch_array)
 
    # By default the new regularizer's weights will be the mean of the old weights
    new_weights = fill(mean(old_reg_copy.weights), length(new_batch_array.col_ranges))

    # Set the new regularizer's weight to match the old weight,
    # whenever appropriate
    old_col_intersect, new_col_intersect = keymatch(old_batch_array.col_range_ids,
                                                    new_batch_array.col_range_ids)
    for (i_old, i_new) in zip(old_value_intersect, new_value_intersect)
        new_weights[i_new] = old_reg_copy.weights[i_old] 
    end

    return BatchArrayReg(Tuple(new_weights))
end

# Behaviors on various other types

function reweight_eb!(reg::BatchArrayReg, bs::BatchShift; mixture_p=1.0)
    reweight_eb!(reg, bs.theta; mixture_p=mixture_p)
end

function reweight_eb!(reg::BatchArrayReg, bs::BatchScale; mixture_p=1.0)
    reweight_eb!(reg, bs.logdelta; mixture_p=mixture_p)
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


####################################################
# Regularizer for `ViewableComposition` structs
####################################################

mutable struct SequenceReg
    regs::Tuple
end

@functor SequenceReg
Flux.trainable(sr::SequenceReg) = (regs=sr.regs)

function (reg::SequenceReg)(seq)
    return sum(map((f,x)->f(x), reg.regs, seq.layers))
end


function construct_layer_reg(feature_views, batch_dict, lambda_layer)

    # Start with the regularizers for logsigma and mu
    # (the column parameters)
    regs = Any[x->0.0, x->0.0, x->0.0, x->0.0]
    if feature_views != nothing
        regs[1] = GroupRegularizer(feature_views; weight=lambda_layer)
        regs[2] = GroupRegularizer(feature_views; weight=lambda_layer)
    end
 
    # If a batch_dict is provided, add regularizers for
    # logdelta and theta (the batch parameters)
    if batch_dict != nothing
        regs[2] = BatchArrayReg(feature_views; weight=lambda_layer)
        regs[4] = BatchArrayReg(feature_views; weight=lambda_layer)
    end

    return SequenceReg(Tuple(regs))    
end

function reweight_eb!(sr::SequenceReg, vc::ViewableComposition; mixture_p=1.0)
    for (r, l) in zip(sr.regs, vc.layers)
        reweight_eb!(r,l; mixture_p=mixture_p)
    end
end

function set_reg!(sr::SequenceReg, idx::Int, reg)
    sr.regs = (sr.regs[1:idx-1]...,
               reg,
               sr.regs[idx+1:end]...)
end

# Pure functions don't have adjustable weights
function reweight_eb!(f::Function, z; mixture_p=1.0)
    return 
end


#####################################################
# A struct for temporarily nullifying a regularizer
#####################################################

mutable struct FrozenRegularizer
    reg 
end

@functor FrozenRegularizer
Flux.trainable(fr::FrozenRegularizer) = ()

function (fr::FrozenRegularizer)(args...)
    return 0.0
end


function ChainRulesCore.rrule(fr::FrozenRegularizer, args...)

    function FrozenRegularizer_pullback(loss_bar)
        return NoTangent(), ZeroTangent()
    end

    return 0.0, FrozenRegularizer_pullback 
end


function freeze_reg!(sr::SequenceReg, idx::Integer)
    if !isa(sr.regs[idx], FrozenRegularizer)
        sr.regs = (sr.regs[1:idx-1]..., 
                   FrozenRegularizer(sr.regs[idx]),
                   sr.regs[idx+1:end]...)
    end
end


function freeze_reg!(sr::SequenceReg, idx::AbstractVector)
    for i in idx
        freeze_reg!(sr, i)
    end
end

function unfreeze_reg!(sr::SequenceReg, idx::Integer)
    if isa(sr.regs[idx], FrozenRegularizer)
        sr.regs = (sr.regs[1:idx-1]..., 
                   sr.regs[idx].reg,
                   sr.regs[idx+1:end]...)
    end
end

function unfreeze_reg!(sr::SequenceReg, idx::AbstractVector)
    for i in idx
        unfreeze_reg!(sr, i)
    end
end


