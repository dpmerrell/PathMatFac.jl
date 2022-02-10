
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
                        feature_assays::Vector{T}) where T where U
        
    return assemble_model(pathway_sif_data,  
                          sample_ids, sample_conditions,
                          sample_batch_dict,
                          feature_genes, feature_assays)

end


function Base.:(==)(model_a::MultiomicModel, model_b::MultiomicModel)
    for fn in fieldnames(MultiomicModel)
        if !(getproperty(model_a, fn) == getproperty(model_b, fn)) 
            return false
        end
    end
    return true
end


