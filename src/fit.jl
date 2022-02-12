
import ScikitLearnBase: fit!


function augment_data_matrix(omic_matrix, 
                             original_samples, internal_samples,
                             original_features, internal_features)

    feat_idx_vec, aug_feat_idx_vec = keymatch(original_features, internal_features)
    sample_idx_vec, aug_sample_idx_vec = keymatch(original_samples, internal_samples)

    M = length(internal_samples)
    N = length(internal_features)
    result = fill(NaN, M, N) 

    for (f_idx, aug_f_idx) in zip(feat_idx_vec, aug_feat_idx_vec)
        result[aug_sample_idx_vec, aug_f_idx] .= omic_matrix[sample_idx_vec, f_idx] 
    end

    return result
end



function fit!(model::MultiomicModel, X::AbstractMatrix; kwargs...)

    orig_features = collect(zip(model.feature_genes, model.feature_assays))
    internal_features = collect(zip(model.internal_feature_genes, 
                                    model.internal_feature_assays))

    internal_X = augment_data_matrix(X, model.sample_ids, 
                                        model.internal_sample_ids,
                                        orig_features, 
                                        internal_features)

    fit!(model.matfac, internal_X; kwargs...)

end


