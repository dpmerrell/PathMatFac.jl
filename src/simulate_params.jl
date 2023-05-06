

###############################################################
# Factor param simulators
###############################################################

function simulate_params!(X::AbstractMatrix, reg::Union{GroupRegularizer,L2Regularizer}; sample_conditions=nothing,
                                                                                         kwargs...)
    K, N = size(X)
    X .= randn_like(X)
end


function simulate_params!(X::AbstractMatrix, reg::ARDRegularizer; ard_alpha=1.05,
                                                                  ard_beta=0.05,
                                                                  kwargs...)
    K, N = size(X)
    gamma_dist = Gamma(ard_alpha, 1/ard_beta)
    tau = rand(gamma_dist, K, N)
    X .= randn_like(X) ./ sqrt.(tau) 
end


function corrupt_S(S::AbstractMatrix, S_add_corruption, S_rem_corruption)
    S_new = convert(Matrix{Float32}, S)
    L, N = size(S_new)
    for l=1:L
        cur_set = Int[j for j=1:N if S_new[l,j] != 0]
        cur_N = length(cur_set)
        rm_N = Int(round(S_rem_corruption*cur_N))
        to_remove = sample(cur_set, rm_N; replace=false)
        cur_complement = Int[j for j=1:N if S_new[l,j] == 0]
        add_N = Int(round(cur_N*S_add_corruption))
        to_add = sample(cur_complement, add_N; replace=false)

        corrupted_set = union(setdiff(Set(cur_set), Set(to_remove)), Set(to_add))
        S_new[l,:] .= 0
        S_new[l,sort(collect(corrupted_set))] .= 1/sqrt(length(corrupted_set))
    end

    return S_new
end


function simulate_params!(Y::AbstractMatrix, reg::FeatureSetARDReg; ard_alpha=1.05,
                                                                    beta0=0.05,
                                                                    S_add_corruption=0.1,
                                                                    S_rem_corruption=0.1,
                                                                    A_density=0.01,
                                                                    kwargs...)
    # Generate geneset->factor assignments for each view
    for (cr, A_mat, S_mat) in zip(reg.col_ranges, reg.A, reg.S)
        Y_view = view(Y, :, cr)
        K, N = size(Y_view)
        L, N = size(S_mat)
        A_mat .= rand(L,K)
        A_mat .= convert(Matrix{Float32}, (A_mat .<= A_density))
        A_mat .*= abs.(randn_like(A_mat)) 
        # Corrupt the matrix of genesets
        S_mat .= corrupt_S(S_mat, S_add_corruption, S_rem_corruption)

        # Compute the entry-specific betas; taus; and then Y
        beta = beta0 .* (1 .+ transpose(A_mat) * S_mat)
        tau = map(b->rand(Gamma(ard_alpha, 1/b)), beta)
        Y_view .= randn_like(Y_view) ./ sqrt.(tau) 
    end
end


#################################################################
# Layer param simulators
#################################################################

#################################
# Batch scales and shifts
function simulate_params!(bs::BatchShift, reg::BatchArrayReg; b_shift_mean=0.0, b_shift_std=0.25, kwargs...)
    simulate_params!(bs.theta, reg; b_mean=b_shift_mean, between_batch_std=b_shift_std, within_batch_std=0.25, kwargs...)
end

function simulate_params!(bs::BatchScale, reg::BatchArrayReg; b_scale_mean=0.0, b_scale_std=0.25, kwargs...)
    simulate_params!(bs.logdelta, reg; b_mean=b_scale_mean, between_batch_std=b_scale_std, within_batch_std=0.25, kwargs...)
end

function simulate_params!(bs::BatchArray, reg::BatchArrayReg; b_mean=0.0, between_batch_std=0.95, within_batch_std=0.05, kwargs...)
    for v in bs.values
        centers = randn_like(v, size(v,1)).*between_batch_std .+ b_mean
        v .= centers .+ (randn_like(v) .* within_batch_std) 
    end
end


######################################
# Noise model param simulators
######################################

function simulate_params!(nm::Union{MF.NormalNoise,MF.SquaredHingeNoise}; kwargs...)
end

function simulate_params!(nm::MF.OrdinalSqHingeNoise; kwargs...)
    nm.ext_thresholds .*= 2.0
end

function simulate_params!(nm::MF.CompositeNoise; kwargs...)
    for n in nm.noises
        simulate_params!(n; kwargs...)
    end
end

################################
# Column scales and shifts 

VIEW_MU_MEAN = Dict("mrnaseq" => 10.0,
                    "methylation" => 0.0, 
                    "cna" => 0.0,
                    "mutation" => -1.5)

VIEW_MU_STD = Dict("mrnaseq" => 2.0,
                   "methylation" => 0.1,
                   "cna" => 0.1,
                   "mutation" => 0.1)

VIEW_LOGSIGMA_MEAN = Dict("mrnaseq" => 0.5,
                          "methylation" => 0.1,
                          "cna" => log(2.0),
                          "mutation" => log(1.0))

VIEW_LOGSIGMA_STD = Dict("mrnaseq" => 0.1,
                         "methylation" => 0.1,
                         "cna" => 0.001,
                         "mutation" => 0.001)

function simulate_params!(cs::ColScale, reg::ColParamReg; feature_views=nothing,
                                                          view_mean=VIEW_LOGSIGMA_MEAN,
                                                          view_std=VIEW_LOGSIGMA_STD, 
                                                          kwargs...)
    simulate_params!(cs.logsigma, reg; feature_views=feature_views, 
                                       view_mean=view_mean,
                                       view_std=view_std,
                                       kwargs...)
end

function simulate_params!(cs::ColShift, reg::ColParamReg; feature_views=nothing,
                                                          view_mean=VIEW_MU_MEAN,
                                                          view_std=VIEW_MU_STD,
                                                          kwargs...)
    simulate_params!(cs.mu, reg; feature_views=feature_views,
                                 view_mean=view_mean,
                                 view_std=view_std,
                                 kwargs...)
end

function simulate_params!(v::AbstractVector, reg::ColParamReg; feature_views=nothing,
                                                               view_mean=VIEW_LOGSIGMA_MEAN,
                                                               view_std=VIEW_LOGSIGMA_STD,
                                                               kwargs...)
    unq_views = unique(feature_views)
    for uv in unq_views
        idx = (feature_views .== uv)
        view_N = sum(idx)
        v[idx] .= randn(view_N).*view_std[uv] .+ view_mean[uv]
    end 
end


############################
# Composition of layers
function simulate_params!(transform::ViewableComposition, reg::SequenceReg; kwargs...)

    for (layer, r) in zip(transform.layers, reg.regs)
        if isa(layer, Function)
            continue
        end
        simulate_params!(layer, r; kwargs...)
    end
end

######################################################################
# The "master" simulate_params! function
######################################################################

function simulate_params!(model::PathMatFacModel; use_sample_conditions=true, sample_condition_var=0.15, kwargs...)

    mf = model.matfac
    K = size(mf.X, 1)

    # Simulate Y
    simulate_params!(mf.Y, mf.Y_reg; kwargs...)

    # Simulate X
    sample_conditions = nothing
    if use_sample_conditions
        sample_conditions = model.sample_conditions
    end
    simulate_params!(mf.X, mf.X_reg; sample_conditions=sample_conditions, 
                                     sample_condition_var=sample_condition_var, kwargs...)

    # Simulate the layer parameters
    simulate_params!(mf.col_transform, mf.col_transform_reg; feature_views=model.feature_views,
                                                             kwargs...)
    mf.col_transform.layers[1].logsigma .-= log(sqrt(K))

    # Simulate any parameters in the noise model
    simulate_params!(mf.noise_model; kwargs...)
end



######################################################################
# Simulate data 
######################################################################

function simulate_data!(data::AbstractMatrix, nm::MF.SquaredHingeNoise; kwargs...)
    idx = data .> 0
    data[idx] .= 1
    data[(!).(idx)] .= 0
end

function simulate_data!(data::AbstractMatrix, nm::MF.OrdinalSqHingeNoise; kwargs...)
    u_idx = data .> nm.ext_thresholds[end-1]
    l_idx = data .<= nm.ext_thresholds[2]
    data[u_idx] .= 3
    data[l_idx] .= 1
    u_idx .= (!).(u_idx)
    l_idx .= (!).(l_idx)
    data[u_idx .& l_idx] .= 2
end

function simulate_data!(data::AbstractMatrix, nm::MF.NormalNoise; noise=0.1, kwargs...)
    data .+= randn_like(data).*noise
end

function simulate_data!(model::PathMatFacModel; noise=0.1,
                                                kwargs...)
    M, N = size(model.data)
    model.data .= MF.forward(model.matfac)
    nm = model.matfac.noise_model

    # Sample from the appropriate distributions and introduce noise.
    for (cr, n) in zip(nm.col_ranges, nm.noises)
        simulate_data!(view(model.data, :, cr), n; noise=noise)
    end
    
end


