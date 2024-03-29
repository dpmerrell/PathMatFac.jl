
export PathMatFacModel

import Base: == 

mutable struct PathMatFacModel

    # Matrix factorization model
    matfac::MatFacModel

    # Data
    data::Union{<:AbstractMatrix,Nothing}

    # Information about data samples
    sample_ids::AbstractVector
    sample_conditions::Union{<:AbstractVector,Nothing}

    # Information about data features
    feature_ids::AbstractVector
    feature_views::AbstractVector

    # The model will generally rearrange the features
    # such that those with the same inverse link functions
    # are put in contiguous blocks. This vector maps
    # from the raw data's index to the model's internal index.
    data_idx::AbstractVector{<:Integer}

end

@functor PathMatFacModel 


#############################################################
# Model assembly internals
#############################################################

function assemble_model(D, K, sample_ids, sample_conditions,
                              feature_ids, feature_views, feature_distributions,
                              batch_dict,
                              feature_sets_dict, featureset_names, feature_graphs, sample_graphs,
                              lambda_X_l2, lambda_X_condition, lambda_X_graph, 
                              lambda_Y_l2, lambda_Y_selective_l1, lambda_Y_graph,
                              Y_ard, Y_feature_set_ard, alpha0, v0,
                              lambda_layer)

    M, N = size(D)
    
    # If necessary, rearrange features such that 
    # their distributions and views are in contiguous blocks
    data_idx = sortperm(collect(zip(feature_distributions, feature_views)))
    feature_ids .= feature_ids[data_idx]
    feature_views .= feature_views[data_idx]
    feature_distributions .= feature_distributions[data_idx]
    D .= D[:,data_idx]

    # Construct the column layers and their regularizer
    col_layers = construct_model_layers(feature_views, batch_dict) 
    layer_reg = construct_layer_reg(feature_views, batch_dict, col_layers, lambda_layer) 

    # Construct regularizers for X and Y
    X_reg = construct_X_reg(K, M, sample_ids, sample_conditions, sample_graphs, 
                            lambda_X_l2, lambda_X_condition, lambda_X_graph,
                            Y_ard, Y_feature_set_ard)
    Y_reg = construct_Y_reg(K, N, feature_ids, feature_views, feature_sets_dict, feature_graphs,
                            lambda_Y_l2, lambda_Y_selective_l1, lambda_Y_graph,
                            Y_ard, Y_feature_set_ard, featureset_names, alpha0, v0)

    # Construct MatFacModel
    matfac = MatFacModel(M, N, K, feature_distributions;
                         col_transform=col_layers,
                         X_reg=X_reg, Y_reg=Y_reg, 
                         col_transform_reg=layer_reg)

    # Construct the PathMatFacModel
    model = PathMatFacModel(matfac, D, 
                            sample_ids, sample_conditions, 
                            feature_ids, feature_views,
                            data_idx)

    return model
end


######################################################
# Main constructor
######################################################
"""
    PathMatFacModel(D; K=10)

    Construct a PathMatFacModel matrix factorization model for dataset D.
"""
function PathMatFacModel(D::AbstractMatrix{<:Real};
                         K::Integer=10,
                         sample_ids::Union{<:AbstractVector,Nothing}=nothing, 
                         sample_conditions::Union{<:AbstractVector,Nothing}=nothing,
                         feature_ids::Union{<:AbstractVector,Nothing}=nothing, 
                         feature_views::Union{<:AbstractVector,Nothing}=nothing,
                         feature_distributions::Union{<:AbstractVector,Nothing}=nothing,
                         batch_dict::Union{<:AbstractDict,Nothing}=nothing,
                         sample_graphs::Union{<:AbstractVector,Nothing}=nothing,
                         feature_sets_dict::Union{<:AbstractDict,Nothing}=nothing,
                         featureset_names::Union{<:AbstractDict,Nothing}=nothing,
                         feature_graphs::Union{<:AbstractVector,Nothing}=nothing,
                         lambda_X_l2::Union{Real,Nothing}=nothing,
                         lambda_X_condition::Union{Real,Nothing}=1.0,
                         lambda_X_graph::Union{Real,Nothing}=1.0, 
                         lambda_Y_l2::Union{Real,Nothing}=1.0,
                         lambda_Y_selective_l1::Union{Real,Nothing}=nothing,
                         lambda_Y_graph::Union{Real,Nothing}=nothing,
                         lambda_layer::Union{Real,Nothing}=1.0,
                         Y_ard::Bool=false,
                         Y_fsard::Bool=false,
                         fsard_alpha0::Number=Float32(1.001),
                         fsard_v0::Number=Float32(0.8)) 
      
    ################################################
    # Validate input
    M, N = size(D)

    # Set the latent dimension from keyword args
    if feature_graphs != nothing
        K = length(feature_graphs)
        if sample_graphs != nothing
            K_sample = length(sample_graphs)
            @assert K == K_sample "`sample_graphs` and `feature_graphs` must have equal length; or one of them must be nothing"
        end
    else
        if sample_graphs != nothing
            K = length(sample_graphs)
        end
    end

    # Sample IDs
    if sample_ids != nothing
        @assert length(sample_ids) == length(unique(sample_ids)) "`sample_ids` must be unique"
        @assert length(sample_ids) == M "`sample_ids` must be nothing or have length equal to size(D,1)"
    else
        sample_ids = collect(1:M)
    end

    # Sample conditions
    if sample_conditions != nothing
        @assert length(sample_conditions) == M "`sample_conditions` must be nothing or have length equal to size(D,1)"
        @assert is_contiguous(sample_conditions) "`sample_conditions` must be contiguous; I.e., samples must be grouped by condition."
    end
    
    # Feature IDs 
    if feature_ids != nothing
        @assert length(feature_ids) == length(unique(feature_ids)) "`feature_ids` must be left default, or set to a vector of unique identifiers"
        @assert length(feature_ids) == N "`feature_ids` must have length equal to dim(D,2)"
    else
        feature_ids = collect(1:N)
    end

    # Batch Dictionary
    if batch_dict != nothing
        @assert feature_views != nothing "`feature_views` must be provided whenever `batch_dict` is provided"
        @assert sample_conditions != nothing "`sample_conditions` must be provided whenever `batch_dict` is provided"
        @assert issubset(Set(keys(batch_dict)), Set(unique(feature_views))) "The `batch_dict` keys must be a subset of `feature_views`"
        for v in values(batch_dict)
            @assert length(v) == M "Each value of `batch_dict` must be a vector of length size(D,1)"
        end
    end

    # Feature views
    if feature_views != nothing
        @assert length(feature_views) == N "`feature_views` must be nothing or have length equal to size(D,2)"
    else
        feature_views = ones(Int64,N)
    end

    # Feature distributions
    if feature_distributions != nothing
        @assert length(feature_distributions) == N "`feature_distributions` must (a) be nothing or have length equal to size(D,2)"
        all_distributions = Set(VALID_LOSSES)
        @assert all(map(x->in(x,all_distributions), feature_distributions)) string("Each entry of `feature_distributions` must be one of ", all_distributions)
    else
        feature_distributions = fill("normal", N)
    end

    # Check whether we're running feature set ARD
    if Y_fsard
        @assert feature_sets_dict != nothing "`feature_sets_dict` must be provided whenever `Y_fsard` is true."
    end 
    

    ###############################
    # Assemble the model
    return assemble_model(D, K, sample_ids, sample_conditions, 
                          feature_ids, feature_views, feature_distributions,
                          batch_dict, feature_sets_dict, featureset_names, feature_graphs, sample_graphs,
                          lambda_X_l2, lambda_X_condition, lambda_X_graph,
                          lambda_Y_l2, lambda_Y_selective_l1, lambda_Y_graph,
                          Y_ard, Y_fsard, fsard_alpha0, fsard_v0, 
                          lambda_layer) 
end


##########################################
# Some extra helper functions
##########################################

PMTypes = Union{PathMatFacModel, NetworkRegularizer, GroupRegularizer,
                SelectiveL1Reg, ColScale, ColShift, BatchScale, BatchShift,
                BatchArray, BatchArrayReg, ViewableComposition, SequenceReg,
                CompositeRegularizer, ARDRegularizer, FeatureSetARDReg}

NoEqTypes = Union{Function,Tuple}


function Base.:(==)(a::T, b::T) where T <: PMTypes
    for fn in fieldnames(T)
        af = getfield(a, fn)
        bf = getfield(b, fn)
        if !(af == bf)
            if !((typeof(af) <: NoEqTypes) & (typeof(bf) <: NoEqTypes))
                return false
            end
        end
    end
    return true
end

function interpret(model::PathMatFacModel; view_names=nothing)
    return interpret(model.matfac.Y_reg; view_names=view_names)
end

