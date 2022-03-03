


function simulate_data(pathway_sif_data, 
                       pathway_names::Vector{String},
                       sample_ids::Vector{String}, 
                       sample_conditions::Vector{String},
                       sample_batch_dict::Dict{T,Vector{U}},
                       feature_genes::Vector{String}, 
                       feature_assays::Vector{T}) where T where U


    model = MultiomicModel(pathway_sif_data, 
                           pathway_names,
                           sample_ids, 
                           sample_conditions,
                           sample_batch_dict,
                           feature_genes, 
                           feature_assay)

    matfac = model.matfac
    theta = BMF.batch_matrix(matfac.theta_values,
                             matfac.feature_batch_ids,
                             matfac.sample_batch_ids)
    log_delta = BMF.batch_matrix(matfac.log_delta_values,
                                 matfac.feature_batch_ids,
                                 matfac.sample_batch_ids)

    sim_params, D_gpu = BMF.simulate_data(matfac.X_reg, matfac.Y_reg,
                                          matfac.mu_reg, matfac.log_sigma_reg,
                                          theta, log_delta, 
                                          matfac.noise_models)

    M = length(model.sample_ids)
    N = length(model.feature_idx)

    D = fill(NaN, M, N)
    D[:, model.feature_idx] .= view(D_gpu, model.internal_sample_idx, 
                                           model.internal_feature_idx)
   
    return D
end


