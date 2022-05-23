
export MultiomicModel

import Base: ==

mutable struct MultiomicModel

    # Matrix factorization model
    matfac::BatchMatFacModel

    # Information about data samples
    sample_ids::Vector{String}
    sample_conditions::Vector{String}

    # Information about data features
    data_genes::Vector{String}
    data_assays::Vector{String}
    used_feature_idx::Vector{Int}

    # Pathway information
    pathway_names::Vector{String}
    pathway_weights::Vector{Float64}

end

@functor MultiomicModel


function MultiomicModel(pathway_sif_data, 
                        pathway_names::Vector{String},
                        sample_ids::Vector{String}, 
                        sample_conditions::Vector{String},
                        data_genes::Vector{String}, 
                        data_assays::Vector{T},
                        sample_batch_dict::Dict{T,Vector{U}};
                        lambda_X::Real=1.0,
                        lambda_Y::Real=1.0,
                        model_features=nothing) where T where U
       
    data_features = collect(zip(data_genes, data_assays))

    return assemble_model(pathway_sif_data, 
                          pathway_names,
                          sample_ids, sample_conditions,
                          sample_batch_dict,
                          data_features,
                          lambda_X, lambda_Y;
                          model_features=model_features)

end

PMTypes = Union{MultiomicModel,NetworkRegularizer,NetworkL1Regularizer,
                BMFLayerReg}

function Base.:(==)(a::T, b::T) where T <: PMTypes
    for fn in fieldnames(T)
        if !(getfield(a, fn) == getfield(b, fn))
            return false
        end
    end
    return true
end



