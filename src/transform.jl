

export transform


function transform(model::PathMatFacModel, D::AbstractMatrix;
                   use_gpu::Bool=true,
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

    # Construct a new model around the new dataset
    old_data = model.data
    model.data = nothing
    new_model = model[:,old_idx]
    new_data = D[:, new_idx]
    new_model.data = new_data
    new_model.feature_ids = feature_ids[new_idx]
    new_model.feature_views = feature_views[new_idx]

    new_model.matfac.X = similar(model.matfac.X, K, M_new) 
    new_model.matfac.X .= 0 
    
    # Construct a new BatchShift layer and BatchArrayReg (if applicable).
    if batch_dict != nothing
        new_layer = BatchShift(feature_views, batch_dict)
        new_reg = construct_new_batch_reg!(model.matfac.col_transform.layers[end],
                                           model.matfac.col_trans_reg.regs[end])
        new_model.matfac.col_trans_reg.regs = (new_model.matfac.col_trans_reg.regs[1:end-1]...,
                                              )
    end

    # Construct a new GroupRegularizer for sample conditions
    # (if applicable)
    if sample_conditions != nothing
        new_model.matfac.X_reg = construct_new_group_reg(sample_conditions,
                                                         model.matfac.X_reg,
                                                         model.matfac.X)
    end

    ##################################################
    # Fit the model to the new data.
    ##################################################
    new_model = gpu(new_model)
    opt = construct_optimizer(new_model, lr)

    # Fit the new BatchShift, in isolation
    if batch_dict != nothing
        # Freeze the other layers
        freeze_layer!(new_model.matfac.col_transform, 1:3)
        mf_fit!(new_model; opt=opt, max_epochs=500, update_col_transform=true)
    end

    # Update the new X 
    mf_fit!(new_model; update_X=true, opt=opt, fit_kwargs...)

    # Finally: update them both, jointly 
    mf_fit!(new_model; update_X=true, update_col_transform=true,
                       opt=opt, fit_kwargs...)

    new_model = cpu(new_model)
    unfreeze_layer!(new_model.matfac.col_transform, 1:3)

    #############################################################
    # Restore the model as it was, and assemble the return values
    #############################################################

    # Restore the original data to the original model
    model.data = old_data

    return new_model 
end



