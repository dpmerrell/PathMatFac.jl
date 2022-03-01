
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
    feature_idx::Vector{Int}
    feature_genes::Vector{String}
    feature_assays::Vector{String}

    # Information about internal features
    internal_feature_idx::Vector{Int}
    internal_feature_genes::Vector{String}
    internal_feature_assays::Vector{String}

    # Pathway information
    pathway_names::Vector{String}
    pathway_weights::Vector{Float64}

end


function MultiomicModel(pathway_sif_data, 
                        pathway_names::Vector{String},
                        sample_ids::Vector{String}, 
                        sample_conditions::Vector{String},
                        sample_batch_dict::Dict{T,Vector{U}},
                        feature_genes::Vector{String}, 
                        feature_assays::Vector{T};
                        lambda_X::Real=0.1,
                        lambda_Y::Real=0.1) where T where U
        
    return assemble_model(pathway_sif_data, 
                          pathway_names,
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
        return view(model.matfac.X, :, model.internal_sample_idx)
    elseif sym == :Y
        return view(model.matfac.Y, :, model.internal_feature_idx)
    elseif sym == :mu
         return view(model.matfac.mu, model.internal_feature_idx)
    elseif sym == :log_sigma
         return view(model.matfac.log_sigma, model.internal_feature_idx)
    elseif sym == :log_delta
        return BMF.batch_matrix(model.matfac.log_delta_values,
                                model.matfac.sample_batch_ids,
                                model.matfac.feature_batch_ids)

    elseif sym == :theta
        return BMF.batch_matrix(model.matfac.theta_values,
                                model.matfac.sample_batch_ids,
                                model.matfac.feature_batch_ids)
    else
        return getfield(model, sym)
    end
end


