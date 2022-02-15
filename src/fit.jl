
import ScikitLearnBase: fit!


function augment_data_matrix(omic_matrix, 
                             original_samples, internal_samples,
                             original_features, internal_features)

    feat_idx_vec, aug_feat_idx_vec = keymatch(original_features, internal_features)
    sample_idx_vec, aug_sample_idx_vec = keymatch(original_samples, internal_samples)

    M = length(internal_samples)
    N = length(internal_features)
    result = fill(NaN, M, N) 

    n = length(sample_idx_vec)
    Threads.@threads for i=1:n 
        @inbounds s_idx = sample_idx_vec[i]
        @inbounds aug_s_idx = sample_idx_vec[i]
        @inbounds result[aug_s_idx, aug_feat_idx_vec] .= omic_matrix[s_idx, feat_idx_vec] 
    end

    return result
end



function fit!(model::MultiomicModel, D::AbstractMatrix; kwargs...)

    orig_features = collect(zip(model.feature_genes, model.feature_assays))
    internal_features = collect(zip(model.internal_feature_genes, 
                                    model.internal_feature_assays))

    internal_D = augment_data_matrix(D, model.sample_ids, 
                                        model.internal_sample_ids,
                                        orig_features, 
                                        internal_features)


    fit!(model.matfac, internal_D; kwargs...)

end


