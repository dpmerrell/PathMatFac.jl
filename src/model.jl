
export MultiomicModel

import Base: == 

mutable struct MultiomicModel

    # Matrix factorization model
    matfac::MatFacModel

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

@functor MultiomicModel (matfac,)


function MultiomicModel(pathway_sif_data, 
                        pathway_names::Vector{String},
                        sample_ids::Vector{String}, 
                        sample_conditions::Vector{String},
                        data_genes::Vector{String}, 
                        data_assays::Vector{T},
                        sample_batch_dict::Dict{T,Vector{U}};
                        lambda_X::Real=1.0,
                        lambda_Y::Real=1.0,
                        lambda_layer::Real=0.1,
                        model_features=nothing) where T where U
       
    data_features = collect(zip(data_genes, data_assays))

    return assemble_model(pathway_sif_data, 
                          pathway_names,
                          sample_ids, sample_conditions,
                          sample_batch_dict,
                          data_features,
                          lambda_X, lambda_Y;
                          lambda_layer=lambda_layer,
                          model_features=model_features)

end

PMTypes = Union{MultiomicModel,NetworkRegularizer,NetworkL1Regularizer, ClusterRegularizer,
                PMLayers,PMLayerReg,ColScale,ColShift,BatchScale,BatchShift,
                BatchArray,BatchArrayReg}

NoEqTypes = Function

function Base.:(==)(a::T, b::T) where T <: PMTypes
    for fn in fieldnames(T)
        af = getfield(a, fn)
        bf = getfield(b, fn)
        if !(af == bf)
            if !((typeof(af) <: NoEqTypes) & (typeof(bf) <: NoEqTypes))
                println(string("(PM) NOT EQUAL: ", fn))
                println(af)
                println(bf)
                return false
            end
        end
    end
    return true
end


