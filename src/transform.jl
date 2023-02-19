

export transform


function transform(model::PathMatFacModel, D::AbstractMatrix;
                   feature_ids::Union{<:AbstractVector,Nothing}=nothing,
                   sample_ids::Union{<:AbstractVector,Nothing}=nothing,
                   sample_conditions::Union{<:AbstractVector,Nothing}=nothing,
                   feature_views::Union{<:AbstractVector,Nothing}=nothing,
                   batch_dict::Union{<:AbstractDict,Nothing}=nothing,
                   lr=0.25, fit_kwargs...)

    K, N = size(model.matfac.Y)
    M_new, N_new = size(D)
    
    ###################################################
    ## Validate input
    ###################################################

    # Manage the *column* attributes.
    # By default, `feature_ids` and `feature_views` are ignored and the columns
    # of new data are assumed to match the columns of training data.
    new_data = nothing
    old_idx = collect(1:N)
    new_idx = collect(1:N_new)
    if feature_ids == nothing
        @assert N_new == N "Columns of D do not match columns of training data. Provide `feature_ids` to ensure they match."
        feature_ids = copy(model.feature_ids)
        if feature_views != nothing
            @assert length(feature_views) == N "`feature_views` must have length == size(D,2) whenever `feature_ids` is not provided"
        end
    # If `feature_ids` are provided, they are matched against the feature IDs
    # of the original training data. If `feature_views` are also provided, then
    # they must match the provided feature_views
    else
        data = similar(D, M_new, N)
        new_data .= NaN
        old_idx, new_idx = keymatch(model.feature_ids, feature_ids)
        if feature_views != nothing
            @assert length(feature_views) == length(feature_ids) "`feature_views` must have same length as `feature_ids`."
        end
    end

    # By default, ignore the sample conditions
    if sample_conditions != nothing
        @assert length(sample_conditions) == M_new "`sample_conditions` must have length == size(D,1)"
    end

    # By default, ignore batch effects in the new data.
    # Otherwise, 
    if batch_dict != nothing
        @assert Set(keys(batch_dict)) == Set(model.feature_ids) "`batch_dict` keys must match the set of `feature_views`"
        for v in values(batch_dict)
            @assert length(v) == M_new "Values of `batch_dict` must have length == size(D,1)"
        end 
    end

    ###################################################
    # Construct a model to transform the new data
    ##################################################

    new_model = model[:,old_idx]
    new_data = D[:, new_idx]

    new_model.matfac.X = similar(model.matfac.X, K, M_new) 
    new_model.matfac.X .= 0 
    #new_model = PathMatFacModel()
    #

    ## Put the new data into the model
    #model.data = new_data
    #model.feature_ids = feature_ids
    ## Swap out the model's X for a new X
    #X_old = model.matfac.X
    #model.matfac.X = similar(X_old, K, M_new)
    #model.matfac.X .= 0

    ## If a batch_dict is provided then match the incoming batches
    ## with those in the model (whenever possible).
    #old_batch = nothing 
    #if length(model.matfac.col_transform.layers) > 2
    #    old_batch = model.matfac.col_transform.layers[end]
    #end
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

    ##################################################
    # Fit the model to the new data.
    ##################################################
    opt = construct_optimizer(model, lr)

    # Fit the new BatchShift, in isolation
    if batch_dict != nothing
        # Freeze the other layers
        freeze_layer!(model.matfac.col_transform, 
                      1:(length(model.matfac.col_transform.layers)-1))
        mf_fit!(model; opt=opt, max_epochs=500, update_col_transform=true)
    end

    # Update the new X 
    mf_fit!(model; update_X=true, opt=opt, fit_kwargs...)

    # Finally: update them both, jointly 
    mf_fit!(model; update_X=true, update_col_transform=true,
                   opt=opt, fit_kwargs...)

    #############################################################
    # Restore the model as it was, and assemble the return values
    #############################################################
    return_values = Any[]

    # Restore the original X 
    X_new = model.matfac.X
    push!(return_values, X_new)
    model.matfac.X = X_old

    # Restore the original X_reg
    if sample_conditions != nothing
        new_X_reg = model.matfac.X_reg
        push!(return_values, new_X_reg)
        if old_X_reg != nothing
            model.matfac.X_reg = old_X_reg
        end
    end

    # If applicable, restore the original batch shift.
    if batch_dict != nothing
        new_batch = model.matfac.col_transform.layers[end]
        push!(return_values, new_batch)
        if old_batch != nothing
            model.matfac.col_transform.layers = (model.matfac.col_transform.layers[1:end-1]..., 
                                                 old_batch) 
        end
    end

    return Tuple(return_values) 
end



