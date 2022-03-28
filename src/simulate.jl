

function simulate_data(pathway_sif_data, 
                       pathway_names::Vector{String},
                       sample_ids::Vector{String}, 
                       sample_conditions::Vector{String},
                       sample_batch_dict::Dict{T,Vector{U}},
                       feature_genes::Vector{String}, 
                       feature_assays::Vector{T},
                       assay_moments_dict::Dict;
                       mu_snr=10.0,
                       delta_snr=10.0,
                       theta_snr=10.0,
                       logistic_mtv=10.0,
                       sample_snr=10.0
                      ) where T where U


    # Translate the pathways and batch ids 
    # into regularization matrices
    model = MultiomicModel(pathway_sif_data, 
                           pathway_names,
                           sample_ids, 
                           sample_conditions,
                           sample_batch_dict,
                           feature_genes, 
                           feature_assays)
    
    unique_feature_batches = unique(model.matfac.feature_batch_ids)
    assay_moments_dict[""] = (0.0, 10.0)
    data_moments_vec = [assay_moments_dict[a] for a in unique_feature_batches]

    # Decompose the means and variances of different column
    # batches -- compute contributions from model parameters
    assay_moment_decomp = BMF.decompose_all_data_signal(data_moments_vec,
                                                        model.matfac.feature_batch_ids,
                                                        model.matfac.feature_noise_models;
                                                        mu_snr=mu_snr,
                                                        delta_snr=delta_snr,
                                                        theta_snr=theta_snr,
                                                        logistic_mtv=logistic_mtv,
                                                        sample_snr=sample_snr)

    # Assemble vectors of batch moments for each model parameter 
    logsigma_moments_vec = []
    mu_moments_vec = []
    logdelta_moments_vec = []
    theta_moments_vec = []
    for a in assay_moment_decomp
        push!(logsigma_moments_vec, a[1])
        push!(mu_moments_vec, a[2])
        push!(logdelta_moments_vec, a[3])
        push!(theta_moments_vec, a[4])
    end

    X_reg = SparseMatrixCSC[SparseMatrixCSC(mat) for mat in model.matfac.X_reg]
    Y_reg = SparseMatrixCSC[SparseMatrixCSC(mat) for mat in model.matfac.Y_reg]
    X_reg = SparseMatrixCSC[SparseMatrixCSC{Float32,Int64}(mat) for mat in X_reg]
    Y_reg = SparseMatrixCSC[SparseMatrixCSC{Float32,Int64}(mat) for mat in Y_reg]

    sim_params = BMF.simulate_params(X_reg, Y_reg,
                                     model.matfac.sample_batch_ids,
                                     model.matfac.feature_batch_ids,
                                     logsigma_moments_vec,
                                     mu_moments_vec,
                                     logdelta_moments_vec,
                                     theta_moments_vec)

    D_gpu = BMF.simulate_data(sim_params, model.matfac.feature_noise_models)

    M = length(model.sample_ids)
    N = length(model.feature_idx)

    D = fill(NaN, M, N)

    D_gpu = D_gpu[:,model.internal_feature_idx]
    D_gpu = D_gpu[model.internal_sample_idx,:]
    D[:, model.feature_idx] .= Matrix(D_gpu)

    return model, sim_params, D
end


