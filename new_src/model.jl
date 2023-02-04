
export PathMatFacModel

import Base: == 

mutable struct PathMatFacModel

    # Matrix factorization model
    matfac::MatFacModel

    # Information about data samples
    sample_ids::AbstractVector
    sample_conditions::AbstractVector

    # Information about data features
    feature_ids::AbstractVector
    feature_views::AbstractVector

    # The model will generally rearrange the features
    # such that those with the same inverse link functions
    # are put in contiguous blocks. This vector maps
    # from the raw data's index to the model's internal index.
    data_to_model::AbstractVector{<:Integer}

end

@functor PathMatFacModel (matfac,)

# TODO make more of these optional
# TODO require data at construction. Constructing
#      an untrained model without a reference dataset
#      makes little sense
function PathMatFacModel(X::AbstractMatrix{<:Real};
                         K::Integer=10,
                         sample_ids::Union{<:AbstractVector,Nothing}=nothing, 
                         sample_conditions::Union{<:AbstractVector,Nothing}=nothing,
                         feature_ids::Union{<:AbstractVector,Nothing}=nothing, 
                         feature_views::Union{<:AbstractVector,Nothing}=nothing,
                         feature_distributions::Union{<:AbstractVector,Nothing}=nothing,
                         batch_map::Union{<:AbstractDict,Nothing}=nothing,
                         lambda_X::Union{Real,Nothing}=nothing,
                         lambda_Y::Union{Real,Nothing}=nothing,
                         lambda_layer::Union{Real,Nothing}=nothing,
                         lambda_l1::Union{Real,Nothing}=nothing,
                         lambda_pathway::Union{Real,Nothing}=nothing) 
      
     
    data_features = collect(zip(data_genes, data_assays))

    return assemble_model(pathway_sif_data, 
                          pathway_names,
                          sample_ids, sample_conditions,
                          sample_batch_dict,
                          data_features,
                          lambda_X, lambda_Y;
                          lambda_layer=lambda_layer,
                          l1_fraction=l1_fraction,
                          model_features=model_features)

end

PMTypes = Union{MultiomicModel,NetworkRegularizer,NetworkL1Regularizer, ClusterRegularizer,
                L1Regularizer, PMLayers,PMLayerReg,ColScale,ColShift,BatchScale,BatchShift,
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


