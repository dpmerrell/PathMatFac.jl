

export transform


function transform(model::PathMatFacModel, D::AbstractMatrix;
                   feature_ids::Union{<:AbstractVector,Nothing},
                   sample_ids::Union{<:AbstractVector,Nothing},
                   sample_conditions::Union{<:AbstractVector,Nothing},
                   batch_dict::Union{<:AbstractDict,Nothing}=nothing,
                   fit_kwargs...)

    K, N = size(model.matfac.Y)
    M_new, N_new = size(D)

    # Ensure that the new data's features are aligned with 
    # the model's training features
    
    model.data = D

    # Swap out the model's X for a new X
    X_old = model.matfac.X
    model.matfac.X = similar(X_old, K, M_new)
    model.matfac.X .= 0

    # If a batch_dict is provided AND the model contains
    # batch parameters, then  match the incoming batches
    # with those in the model (whenever possible). 
    #
    # Collect the relevant values of the batch parameters. 
    #
    # Construct a new BatchShift layer from them.
     

    # If sample conditions are provided, then construct 
    # a new group regularizer for the new X.
    #
    # If the model already contains batch parameters, then match the incoming
    # conditions with those in the model (whenever possible).
    #
    # Collect the relevant means from the condition regularizers.
    # 


    # Fit the model to the new data.
    # Only update the new BatchShift,
    freeze_layer!(model.matfac.col_transform, 1)
    freeze_layer!(model.matfac.col_transform, 2)
    freeze_layer!(model.matfac.col_transform, 3)
    mf_fit!(model; lr=0.25, max_epochs=500, update_col_transform=true)

    
    # Update the new X 
    mf_fit!(model; update_X=true, fit_kwargs...)


    # Update them both, jointly 
    mf_fit!(model; update_X=true, update_col_transform=true,
                   fit_kwargs...)


    # Restore the model as it was.
    # Restore the original X, the original X_reg,
    # and the original BatchShift.
end



