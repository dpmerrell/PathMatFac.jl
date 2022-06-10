
using SparseArrays

#######################################################
# Sampling data for different noise models
#######################################################


function sample(nm::MF.NormalNoise, Z::AbstractMatrix; std=0.1, kwargs...)
    return Z .+ (randn(size(Z)...) .* std)
end


function sample(nm::MF.BernoulliNoise, Z::AbstractMatrix; xor_p=0.01, kwargs...)
    X = rand(size(Z)...)
    X = (X .> Z)
    to_flip = (rand(size(Z)...) .<= xor_p)
    X = xor.(X, to_flip)
    return X
end


function sample(nm::MF.OrdinalNoise, Z::AbstractMatrix; ordinal_std=0.1, kwargs...)

    Z .+= randn(size(Z)...) .* ordinal_std

    N_bins = length(nm.ext_thresholds)-1
    result = similar(Z)
    for i=1:N_bins
        relevant_idx = ((Z .> nm.ext_thresholds[i]) .& (Z .<= nm.ext_thresholds[i+1]))
        result[relevant_idx] .= i
    end

    return result
end


function sample(nm::MF.CompositeNoise, Z::AbstractMatrix; std=0.1, xor_p=0.01, ordinal_std=0.1)

    X = similar(Z)
    for (idx, nm) in zip(nm.col_ranges, nm.noises)
        X[:,idx] .= sample(nm, Z[:,idx]; std=std)
    end

    return X
end


#######################################################
# Generate the model parameters
#######################################################

function simulate_X!(model::MultiomicModel; within_sigma=1.0, between_sigma=1.0)

    K, M = size(model.matfac.X)

    conditions = unique(model.sample_conditions)
    M_conditions = length(conditions)

    condition_centers = randn(K, M_conditions) .* between_sigma

    X = randn(K,M) .* within_sigma

    for (j,cond) in enumerate(conditions)
        X[:, model.sample_conditions .== cond] .+= condition_centers[:,j]
    end

    model.matfac.X = X

end


function sample_normal(precision_matrix::AbstractMatrix)

    N = size(precision_matrix,1)
    z = randn(N)

    x = precision_matrix \ z
    return x
end


function simulate_Y!(model::MultiomicModel; average_non_pwy=5.0)

    K, N = size(model.matfac.Y)

    Y = zeros(K,N)
    for k=1:K

        # Unpack the regularizer for this row of Y
        netreg = model.matfac.Y_reg
        AA = netreg.AA[k]
        AB = netreg.AB[k]
        BB = netreg.BB[k]
        non_pwy_idx = netreg.l1_feat_idx[k]

        N_virtual = size(BB,1)
        N_total = N + N_virtual

        # Construct the precision matrix
        precision = spzeros(N_total, N_total)
        precision[1:N,1:N] .= AA
        precision[1:N, N+1:N_total] .= AB
        precision[N+1:N_total, 1:N] .= transpose(AB)
        precision[N+1:N_total, N+1:N_total] .= BB

        # Sample from the MVN defined by the precision matrix
        y = sample_normal(precision)
        y = y[1:N]

        # Set most of the non-pathway members to zeros
        n_non_pwy = sum(non_pwy_idx)
        p_non_pwy = average_non_pwy/n_non_pwy
        to_flip = (rand(N) .<= p_non_pwy)
        mask = (!).(non_pwy_idx) .| to_flip
        y .*= mask

        Y[k,:] = y

    end

    model.matfac.Y = Y

end


function simulate_col_params!(model::MultiomicModel; std=0.01)

    # TODO: Allow assays to determine these parameters

    N = length(model.matfac.col_transform.cscale.logsigma)

    model.matfac.col_transform.cscale.logsigma = randn(N).*std
    model.matfac.col_transform.cshift.mu = randn(N).*std

end


function simulate_batch_params!(model::MultiomicModel; std=0.01)

    logdelta_v = model.matfac.col_transform.bscale.logdelta.values
    theta_v = model.matfac.col_transform.bshift.theta.values

    for i=1:length(logdelta_v)

        M_batch = length(logdelta_v[i])
        logdelta_v[i] .= randn(M_batch).*std
        theta_v[i] .= randn(M_batch).*std
    end

end


########################################################
# The main data simulation function
########################################################

function simulate_data(pathway_sif_data, pathway_names,
                       sample_ids, sample_conditions,
                       data_genes, data_assays,  
                       sample_batch_dict) 

    # Construct the model object
    model = MultiomicModel(pathway_sif_data, pathway_names,
                           sample_ids, sample_conditions,
                           data_genes, data_assays,
                           sample_batch_dict) 

    # Generate the model parameters
    simulate_X!(model)
    simulate_Y!(model)
    simulate_col_params!(model)
    simulate_batch_params!(model)

    # Run the model in forward mode
    Z = model.matfac()

    # Sample random values
    Z = sample(model.matfac.noise_model, Z)

    # Rearrange the columns so they match the original data's ordering
    Z[:, model.used_feature_idx] .= Z
    return Z
end


