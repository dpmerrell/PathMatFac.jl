
import Base: ==

# By default, `reorder_reg!` does nothing.
function reorder_reg!(reg, p) end

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

function reweight_eb!(reg::L2Regularizer, X::AbstractMatrix; mixture_p=Float32(1.0))
    #row_vars = vec(var(X, dims=2))
    #max_var = maximum(row_vars)
    #reg.weights .= (mixture_p / max_var)
    F = svd(X)
    S = F.S
    max_var = S[1]^2
    reg.weights .= mixture_p / max_var
end 

function reweight_eb!(reg::L2Regularizer, x::AbstractVector; mixture_p=Float32(1.0))
    reweight_eb!(reg, transpose(x); mixture_p=mixture_p)
end 

function reorder_reg!(reg::L2Regularizer, p)
    reg.weights .= reg.weights[p]
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

function reorder_reg!(reg::L1Regularizer, p)
    reg.weights .= reg.weights[p]
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

function reorder_reg!(reg::SelectiveL1Reg, p)
    reg.l1_idx .= reg.l1_idx[p,:]
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

    loss = 0
    K,M = size(X)
    for k=1:K
        # Network-regularization
        xAB = transpose(transpose(X[k,:])*nr.AB[k])
        nr.x_virtual[k] .= - cg(nr.BB[k], xAB, nr.x_virtual[k])[1]

        net_loss = 0
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

    loss = 0
    for k=1:K
        # Network-regularization
        # Parts of the gradients
        xAA[k,:] .= vec(transpose(X[k,:])*nr.AA[k])
        xAB = vec(transpose(X[k,:])*nr.AB[k])
        nr.x_virtual[k] .= - cg(nr.BB[k], xAB, nr.x_virtual[k])[1]
        ABu[:,k] .= nr.AB[k]*nr.x_virtual[k]
        BBu = nr.BB[k]*nr.x_virtual[k]

        # Full loss computation
        net_loss = 0
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

function reorder_reg!(reg::NetworkRegularizer, p)

    reg.AA = reg.AA[p]
    reg.AB = reg.AB[p]
    reg.BB = reg.BB[p]
    reg.x_virtual = reg.x_virtual[p]
    reg.cur_weights .= reg.cur_weights[p]
    return
end 

##########################################################
# Group Regularizer -- meant to regularize X when 
# the samples share conditions
##########################################################

mutable struct GroupRegularizer
    group_labels::AbstractVector
    group_idx::Tuple
    group_weights::Tuple # Tuple of vectors -- row-and-group-specific weights
end

@functor GroupRegularizer

function GroupRegularizer(group_labels::AbstractVector; weight=1.0, K=1)
    unq_labels = unique(group_labels)
    idx = ids_to_ranges(group_labels)
    n_groups = length(unq_labels)
    weights = [fill(weight, K) for k=1:n_groups]
    return GroupRegularizer(unq_labels, Tuple(idx), Tuple(weights)) #, Tuple(centers))
end


function construct_new_group_reg(new_group_labels::AbstractVector,
                                 old_regularizer::GroupRegularizer, 
                                 old_array::AbstractMatrix; 
                                 mixture_p=1.0)

    K = size(old_array, 1)

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
    new_weights[new_intersect_idx] .= collect(old_reg_copy.group_weights[old_intersect_idx])

    ## Construct a vector of group centers for the new regularizer.
    ## By default, set it to the mean of the old centers.
    #expected_center = sum(old_reg_copy.group_centers) ./ length(old_reg_copy.group_centers)
    #new_centers = [copy(expected_center) for _=1:length(unq_new_labels)]
    #for (ni, oi) in zip(new_intersect_idx, old_intersect_idx)
    #    new_centers[ni] .= old_reg_copy.group_centers[oi]
    #end

    # Finally: construct the sparse index matrix for the new regularizer
    new_group_idx = ids_to_ranges(new_group_labels)

    # Return the completed regularizer
    return GroupRegularizer(unq_new_labels, 
                            Tuple(new_group_idx), 
                            Tuple(new_weights)) 
end


function reweight_eb!(gr::GroupRegularizer, X::AbstractMatrix; mixture_p=Float32(1.0))
    #new_vars = map(idx -> mean(view(X,:,idx).^2, dims=2), gr.group_idx)
    #max_vars = map(v -> fill(maximum(v), size(v)), new_vars)
    #new_weights = map(v -> Float32(1e-1 .+ 0.5) ./ (Float32(1e-1) .+ Float32(0.5).*v), max_vars)
    #if isa(X, CuArray)
    #    new_weights = map(v -> gpu(v), new_weights)
    #end
    # 
    #gr.group_weights = new_weights
    K = size(X,1)
    new_weights = []
    for gidx in gr.group_idx
        X_v = view(X, :, gidx)
        F = svd(X_v)
        s = F.S
        v_max = s[1]^2
        push!(new_weights, fill(Float32(mixture_p/v_max), K))
    end
    gr.group_weights = Tuple(new_weights)
end


function (gr::GroupRegularizer)(X::AbstractMatrix)
    return Float32(0.5)*sum(map((w, idx)->sum( w.*view(X,:,idx).^2), 
                                gr.group_weights, gr.group_idx
                               ) 
                           )
end


function ChainRulesCore.rrule(gr::GroupRegularizer, X::AbstractMatrix)

    diffs = zero(X)
    for (w,cr) in zip(gr.group_weights, gr.group_idx)
        X_view = view(X,:,cr)
        diffs_view = view(diffs,:,cr)
        diffs_view .= w.*X_view
    end
    loss = Float32(0.5)*sum(diffs.*X)

    function groupreg_pullback(loss_bar)
        return NoTangent(), loss_bar.*diffs
    end

    return loss, groupreg_pullback
end


function reorder_reg!(reg::GroupRegularizer, p)
    reg.group_weights = map(w -> w[p], reg.group_weights)
    return
end

function (gr::GroupRegularizer)(layer::FrozenLayer)
    return 0
end

##############################################
# ColParamReg
##############################################

mutable struct ColParamReg
    col_ranges::Tuple
    weights::Tuple
    centers::Tuple
end


@functor ColParamReg
Flux.trainable(cpr::ColParamReg) = ()


function ColParamReg(feature_views; weight=1.0, center=0)
    col_ranges = Tuple(ids_to_ranges(feature_views))
    n_range = length(col_ranges)
    weights = Tuple(fill(weight, n_range))
    centers = Tuple(fill(center, n_range))
    return ColParamReg(col_ranges, weights, centers)
end


function (cpr::ColParamReg)(v::AbstractVector)
    return 0.5*sum(map((c,w,r)->w*sum((v[r] .- c).^2), cpr.centers,
                                                       cpr.weights,
                                                       cpr.col_ranges)
                  )
end


function reweight_eb!(cpr::ColParamReg, v::AbstractVector; mixture_p=1.0)
    new_centers = map(r->mean(v[r]), cpr.col_ranges)
    new_vars = map(r->var(v[r]), cpr.col_ranges)
    new_weights = mixture_p .* Float32(1e-1 + 0.5) ./ (Float32(1e-1) .+ Float32(0.5).*new_vars)
    
    cpr.centers = new_centers
    cpr.weights = new_weights
end


function (cpr::ColParamReg)(cs::ColShift)
    return cpr(cs.mu)
end

function (cpr::ColParamReg)(cs::ColScale)
    return cpr(cs.logsigma)
end

function (cpr::ColParamReg)(fl::FrozenLayer)
    return 0
end

## Define behaviors on various types
function reweight_eb!(cpr::ColParamReg, cs::ColShift; mixture_p=1.0)
    reweight_eb!(cpr, cs.mu; mixture_p=mixture_p)
end

function reweight_eb!(cpr::ColParamReg, cs::ColScale; mixture_p=1.0)
    reweight_eb!(cpr, cs.logsigma; mixture_p=mixture_p)
end


###########################################
# Automatic Relevance Determination (ARD) 
###########################################

mutable struct ARDRegularizer
    alpha::Tuple
    beta::Tuple
    col_ranges::Tuple
    weight::Number
end

@functor ARDRegularizer

function ARDRegularizer(column_groups::AbstractVector; alpha=Float32(1.001), 
                                                       beta=Float32(0.001),
                                                       weight=1.0)
    col_ranges = ids_to_ranges(cpu(column_groups))
    n_ranges = length(col_ranges)
    alpha = Tuple(fill(alpha, n_ranges))
    beta = Tuple(fill(beta, n_ranges))
    return ARDRegularizer(alpha, beta, Tuple(col_ranges), weight)
end


function (ard::ARDRegularizer)(X::AbstractMatrix)
    buffer = zero(X)
    result = 0 
    for (a,b,cr) in zip(ard.alpha, ard.beta, ard.col_ranges)
        X_v = view(X, :, cr)
        buffer[:,cr] .= 1 .+ (0.5/b).*(X_v.*X_v)
        buffer[:,cr] .= log.(buffer[:,cr])
        result += (0.5 + a)*sum(view(buffer,:,cr))
    end
    return sum(result)
    #b = 1 .+ (0.5/ard.beta).*(X.*X)
    #return (0.5 + ard.alpha)*sum(log.(b))
end


function ChainRulesCore.rrule(ard::ARDRegularizer, X::AbstractMatrix)
    
    buffer = zero(X)
    result = 0 
    for (b,cr) in zip(ard.beta, ard.col_ranges)
        X_v = view(X, :, cr)
        buffer[:,cr] .= 1 .+ (0.5/b).*(X_v.*X_v)
    end

    function ard_pullback(loss_bar)
        X_bar = similar(X)
        for (a, b, cr) in zip(ard.alpha, ard.beta, ard.col_ranges)
            Xb_v = view(X_bar, :, cr)
            Xb_v .= (loss_bar/b)*(0.5 + a) .* view(X,:,cr) ./ view(buffer,:,cr)
            #buffer[:,cr] .= 1 .+ (0.5/b).*(Xb_v.*Xb_v)
        end
        return NoTangent(), X_bar 
    end

    for (a, cr) in zip(ard.alpha, ard.col_ranges)
        result += (0.5 + a)*sum(log.(view(buffer,:,cr)))
    end

    return result, ard_pullback
end


function reweight_eb!(reg::ARDRegularizer, X::AbstractMatrix)

    alpha_mom = fill(0.001, length(reg.alpha))
    beta_mom = fill(0.001, length(reg.beta)) 
    #alpha_mom = zeros(length(reg.alpha))
    #beta_mom = zeros(length(reg.beta))

    #for (i,cr) in enumerate(reg.col_ranges)
    #    X_v = view(X, :, cr)
    #    tau_pm = Float32(1e-6 + 0.5) ./ (Float32(1e-6) .+ Float32(0.5) .* (X_v.*X_v))
    #    tau_mean_vec = mean(tau_pm, dims=2)
    #    max_mean_idx = argmax(vec(tau_mean_vec))
    #    tau_mean = tau_mean_vec[max_mean_idx]
    #    tau_var = var(tau_pm[max_mean_idx,:])
    #    
    #    beta_mom[i] = tau_mean / tau_var
    #    alpha_mom[i] = tau_mean*tau_mean/tau_var
    #end

    reg.alpha = Tuple(alpha_mom)
    reg.beta = Tuple(beta_mom)
end


#######################################################
# Struct representing a weighted sum of regularizers 
#######################################################

mutable struct CompositeRegularizer
    regularizers::Tuple
    mixture_p::Tuple
end

@functor CompositeRegularizer
Flux.trainable(cr::CompositeRegularizer) = (regularizers=cr.regularizers,)


function construct_composite_reg(regs::AbstractVector, mixture_p::AbstractVector)
    if length(regs) == 0
        regs = [ x->0 ]
        mixture_p = [0]
    end
    return CompositeRegularizer(Tuple(regs), Tuple(mixture_p))
end


function reweight_eb!(cr::CompositeRegularizer, X; super_mixture_p=1.0)
    for (reg, p) in zip(cr.regularizers, cr.mixture_p)
        reweight_eb!(reg, X; mixture_p=p*super_mixture_p) 
    end
end


function (cr::CompositeRegularizer)(x)
    return sum(map((f,p)->p*f(x), cr.regularizers, cr.mixture_p))
end

function reorder_reg!(reg::CompositeRegularizer, p)
    for r in reg.regularizers
        reorder_reg!(r, p)
    end
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

    regularizers = Any[x->0, x->0, x->0]
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

function construct_Y_reg(K, N, feature_ids, feature_views, feature_sets_dict, feature_graphs,
                         lambda_Y_l2, lambda_Y_selective_l1, lambda_Y_graph,
                         Y_ard, Y_geneset_ard, featureset_names, alpha0, v0)

    # If either of the ARD flags are set `true`, then
    # they take priority over the other regularizers.
    if Y_geneset_ard
        return construct_featureset_ard(K, feature_ids, feature_views, feature_sets_dict; 
                                        featureset_ids=featureset_names, alpha0=alpha0, v0=v0) 
    end
    if Y_ard
        return ARDRegularizer(feature_views) 
    end

    # If neither ARD flag is set `true`, then 
    # construct a regularizer from any other flags.
    # (Default is *no* regularization) 
    regularizers = Any[x->0, x->0, x->0]
    mixture_p = zeros(3)
    if lambda_Y_l2 != nothing
        regularizers[1] =  GroupRegularizer(feature_views; K=K, weight=lambda_Y_l2)
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

    s = sum(mixture_p)
    if s == 0
        s = 1
    end
    mixture_p ./= s
    return construct_composite_reg(regularizers, mixture_p) 
end


##########################################################
# Construct a regularizer for the first phase of training.
# This prevents the model parameters from 
# taking crazy values during the first phase.
# It's especially important for columns of data that are
# sparse (and therefore admit many possible solutions).
##########################################################

function construct_minimal_regularizer(model; capacity=10^8)

    K,M = size(model.matfac.X)
    N = size(model.matfac.Y, 2)

    # Construct a GroupRegularizer based on the column losses
    group_labels = [string(typeof(n)) for n in model.matfac.noise_model.noises]
    group_idx = deepcopy(model.matfac.noise_model.col_ranges)
    n_groups = length(group_idx)
    group_weights = [zeros_like(model.matfac.Y, K) for _=1:n_groups]
    
    # L2-penalize factors in a way that accounts for K, layer gradients, and the 
    # density/sparseness of the data 
    sigma = exp.(model.matfac.col_transform.layers[1].logsigma)
    for (gidx, gl, gw) in zip(group_idx, group_labels, group_weights)
        M_vec = MF.column_nonnan(view(model.data, :, gidx))
        col_vars = MF.batched_column_nanvar(view(model.data, :, gidx); capacity=capacity)
        map!(v->max(v, Float32(1/M)), col_vars, col_vars)
        gw .= K*mean(sigma[gidx].^2) / (sum(col_vars .* M_vec) / M) 
    end
    reg = GroupRegularizer(group_labels,
                           group_idx,
                           Tuple(group_weights))
    return reg
end


##########################################################
# Regularizer for Batch Array structs
##########################################################

mutable struct BatchArrayReg 
    centers::Tuple # Tuple of vectors -- centers of quadratic regularization 
    weights::Tuple # Tuple of vectors -- strengths of quadratic regularization
end

@functor BatchArrayReg 

function BatchArrayReg(ba::BatchArray; center=0, weight=1.0)
    N_batches = map(v -> size(v,1), ba.values)
    return BatchArrayReg(map( n -> fill(center, n), N_batches),
                         map( n -> fill(weight, n), N_batches)) 
end


function (reg::BatchArrayReg)(ba::BatchArray)
    diffs = map((v,c) -> v.-c, ba.values, reg.centers)
    return 0.5*sum(map((w,d)->sum(w .* d .* d), reg.weights, diffs))
end


function ChainRulesCore.rrule(reg::BatchArrayReg, ba::BatchArray)

    diffs = map((v,c) -> v.-c, ba.values, reg.centers)
    grad = map((w,d)->w.*d, reg.weights, diffs)

    function batcharray_reg_pullback(loss_bar)
        val_bar = loss_bar .* grad
        return ChainRulesCore.NoTangent(),
               ChainRulesCore.Tangent{BatchArray}(values=val_bar)

    end
    lss = 0.5*sum(map((g,d) -> sum(g .* d), grad, diffs))

    return lss, batcharray_reg_pullback
end


function reweight_eb!(reg::BatchArrayReg, A::BatchArray; mixture_p=1.0)

    n_col_range = length(A.col_ranges)    
    new_weights = map(zero, reg.weights)
   
    new_centers = map(v -> vec(mean(v, dims=2)), A.values)
    new_vars = map(v -> vec(var(v, dims=2)), A.values)

    reg.centers = new_centers
    reg.weights = map(v -> mixture_p ./ v, new_vars)
   
    # Set regularizer weights to a finite value
    # whenever they're NaN
    nan_idx = map(w -> (!isfinite).(w), reg.weights)
    cr_sizes = map(length, A.col_ranges)
    for (w,i,s) in zip(reg.weights, nan_idx, cr_sizes)
        w[i] .= (1 + Float32(0.5)*s) # This is the posterior mean of a gamma-distributed precision
                                     # with parameters α = β = 1 (and zero empirical variance). 
    end

end


# Construct a new BatchArrayReg for the `new_batch_array`;
# but let its biases be informed by an `old_batch_array`.
# And mutate `new_batch_array`'s values to agree with those biases.
function construct_new_batch_reg!(new_batch_array::BatchArray, 
                                  old_batch_array_reg::BatchArrayReg,
                                  old_batch_array)
 
    ## Make a copy of the old regularizer, and update its weights
    ## in an empirical Bayes fashion. 
    #old_reg_copy = deepcopy(old_batch_array)
    #reweight_eb!(old_reg_copy, old_batch_array)

    ## Look for intersecting column ranges between the new and old batch array
    #old_col_intersect, new_col_intersect = keymatch(old_batch_array.col_range_ids,
    #                                                new_batch_array.col_range_ids)
    #for (i_old, i_new) in zip(old_col_intersect, new_col_intersect)

    #    old_rowbatch_intersect, new_rowbatch_intersect = keymatch(old_batch_array.row_batch_ids[i_old],
    #                                                              new_batch_array.row_batch_ids[i_new])        

    #    old_row_intersect, 
    #    new_weights[i_new] = old_reg_copy.weights[i_old] 
    #end

    #return BatchArrayReg(Tuple(new_weights))
    return BatchArrayReg(new_batch_array)
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

function (reg::BatchArrayReg)(layer::FrozenLayer)
    return 0
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


function construct_layer_reg(feature_views, batch_dict, layers, lambda_layer)

    # Start with the regularizers for logsigma and mu
    # (the column parameters)
    regs = Any[x->0, x->0, x->0, x->0]
    if feature_views != nothing
        regs[1] = ColParamReg(feature_views; weight=lambda_layer)
        regs[3] = ColParamReg(feature_views; weight=lambda_layer)
    end
 
    # If a batch_dict is provided, add regularizers for
    # logdelta and theta (the batch parameters)
    if batch_dict != nothing
        regs[2] = BatchArrayReg(layers.layers[2].logdelta; weight=lambda_layer)
        regs[4] = BatchArrayReg(layers.layers[4].theta; weight=lambda_layer)
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
    return 0
end


function ChainRulesCore.rrule(fr::FrozenRegularizer, args...)

    function FrozenRegularizer_pullback(loss_bar)
        return NoTangent(), ZeroTangent()
    end

    return 0, FrozenRegularizer_pullback 
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


function reorder_reg!(r::FrozenRegularizer, p)
    reorder_reg!(r.reg, p)
end


