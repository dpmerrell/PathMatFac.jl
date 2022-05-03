
export MultiomicModel

import Base: ==

mutable struct MultiomicModel

    # Matrix factorization model
    matfac::BatchMatFacModel

    # Information about data samples
    sample_ids::Vector{String}
    sample_conditions::Vector{String}

    # Information about data features
    feature_idx::Vector{Int}
    feature_genes::Vector{String}
    feature_assays::Vector{String}

    # Pathway information
    pathway_names::Vector{String}
    pathway_weights::Vector{Float64}

end

@functor MultiomicModel


function MultiomicModel(pathway_sif_data, 
                        pathway_names::Vector{String},
                        sample_ids::Vector{String}, 
                        sample_conditions::Vector{String},
                        feature_genes::Vector{String}, 
                        feature_assays::Vector{T},
                        sample_batch_dict::Dict{T,Vector{U}};
                        lambda_X::Real=1.0,
                        lambda_Y::Real=1.0;
                        model_features=nothing) where T where U
       
    features = collect(zip(feature_genes, feature_assays))
    return assemble_model(pathway_sif_data, 
                          pathway_names,
                          sample_ids, sample_conditions,
                          sample_batch_dict,
                          features,
                          lambda_X, lambda_Y;
                          model_features=model_features)

end

PMTypes = Union{MultiomicModel,NetworkRegularizer}

function Base.:(==)(a::T, b::T) where T <: PMTypes
    for fn in fieldnames(T)
        if !(getfield(a, fn) == getfield(b, fn))
            println(string("NOT EQUAL: ", fn))
            return false
        end
    end
    return true
end



