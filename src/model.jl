
export MultiomicModel

mutable struct MultiomicModel

    # Matrix factorization model
    matfac::BatchMatFacModel

    # Information about data samples
    sample_ids::Vector{String}
    sample_conditions::Vector{String}

    # Information about internal samples
    internal_sample_idx::Vector{Int}
    internal_sample_ids::Vector{String}

    # Information about data features
    feature_genes::Vector{String}
    feature_assays::Vector{String}

    # Information about internal features
    internal_feature_idx::Vector{Int}
    internal_feature_genes::Vector{String}
    internal_feature_assays::Vector{String}

end


function MultiomicModel(pathway_sif_data,  
                        sample_ids::Vector{String}, 
                        sample_conditions::Vector{String},
                        sample_batch_dict::Dict{T,Vector{U}},
                        feature_genes::Vector{String}, 
                        feature_assays::Vector{T};
                        lambda_X::Real=1.0,
                        lambda_Y::Real=1.0) where T where U
        
    return assemble_model(pathway_sif_data,  
                          sample_ids, sample_conditions,
                          sample_batch_dict,
                          feature_genes, feature_assays,
                          lambda_X, lambda_Y)

end


function Base.:(==)(model_a::MultiomicModel, model_b::MultiomicModel)
    for fn in fieldnames(MultiomicModel)
        if !(getfield(model_a, fn) == getfield(model_b, fn)) 
            return false
        end
    end
    return true
end


function Base.getproperty(model::MultiomicModel, sym::Symbol)

    if sym == :X
        return model.matfac.X[:,model.internal_sample_idx]
    elseif sym == :Y
        return model.matfac.Y[:,model.internal_feature_idx]
    elseif sym == :mu
         return model.matfac.mu[model.internal_feature_idx]
    elseif sym == :sigma
         return exp.(model.matfac.log_sigma[model.internal_feature_idx])
    elseif sym == :delta
        log_delta = BMF.batch_matrix(model.matfac.log_delta_values,
                                     model.matfac.sample_batch_ids,
                                     model.matfac.feature_batch_ids)

        return exp(model.matfac.log_delta)
    elseif sym == :theta
        return BMF.batch_matrix(model.matfac.theta_values,
                                model.matfac.sample_batch_ids,
                                model.matfac.feature_batch_ids)
    else
        return getfield(model, sym)
    end
end


