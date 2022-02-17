
import ScikitLearnBase: fit!


function augment_data_matrix(omic_matrix, 
                             M, sample_idx, internal_sample_idx,
                             N, feature_idx, internal_feature_idx)

    result = fill(NaN, M, N) 

    Threads.@threads for i=1:length(sample_idx) 
        s_idx = sample_idx[i]
        aug_s_idx = internal_sample_idx[i]
        result[aug_s_idx, internal_feature_idx] .= omic_matrix[s_idx, feature_idx] 
    end

    return result
end


function normalize_factors!(model::MultiomicModel)

    Y = model.Y

    Y_norms = sqrt.(sum(Y .* Y; dims=2))
    target_Y_norm = size(Y,2)/100
    model.matfac.Y .*= (target_Y_norm ./ Y_norms)

    model.pathway_weights .= dropdims(Matrix(Y_norms); dims=2) ./ (target_Y_norm) 

end


function fit!(model::MultiomicModel, D::AbstractMatrix; kwargs...)

    orig_features = collect(zip(model.feature_genes, model.feature_assays))
    internal_features = collect(zip(model.internal_feature_genes, 
                                    model.internal_feature_assays))
    
    M = length(model.internal_sample_ids)
    N = length(model.internal_feature_genes)

    n_samples = length(model.sample_ids)

    internal_D = augment_data_matrix(D, M, collect(1:n_samples), 
                                        model.internal_sample_idx,
                                        N,  model.feature_idx, 
                                        model.internal_feature_idx)

    normalize_factors!(model)

    fit!(model.matfac, internal_D; kwargs...)

end


