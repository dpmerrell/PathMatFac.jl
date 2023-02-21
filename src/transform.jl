

export transform


function transform(model::PathMatFacModel, D::AbstractMatrix;
                   use_gpu::Bool=true,
                   feature_ids::Union{<:AbstractVector,Nothing}=nothing,
                   sample_ids::Union{<:AbstractVector,Nothing}=nothing,
                   sample_conditions::Union{<:AbstractVector,Nothing}=nothing,
                   feature_views::Union{<:AbstractVector,Nothing}=nothing,
                   batch_dict::Union{<:AbstractDict,Nothing}=nothing,
                   verbosity=1, lr=0.25, fit_kwargs...)

    K, N = size(model.matfac.Y)
    M_new, N_new = size(D)

    println("Transforming new data...")

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
        if feature_views != nothing
            @assert length(feature_views) == length(feature_ids) "`feature_views` must have same length as `feature_ids`."
        end
        old_idx, new_idx = keymatch(model.feature_ids, feature_ids)
    end

    # By default, ignore the sample conditions
    if sample_conditions != nothing
        @assert length(sample_conditions) == M_new "`sample_conditions` must have length == size(D,1)"
    end

    if sample_ids != nothing
        @assert length(sample_ids) == M_new "`sample_ids` must have length == size(D,1)"
    else
        sample_ids = collect(1:M_new)
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

    # Construct a model around the new dataset
    
    # Data & model
    old_data = model.data
    model.data = nothing
    new_model = deepcopy(model)
    new_data = D[:, new_idx]
    new_model.data = new_data

    # Construct "feature attributes"
    new_model.matfac.Y = new_model.matfac.Y[:,old_idx]
    new_model.matfac.noise_model = model.matfac.noise_model[old_idx]
    new_model.feature_ids = feature_ids[new_idx]
    new_model.feature_views = feature_views[new_idx]

    # Construct "sample attributes"
    new_model.matfac.X = similar(model.matfac.X, K, M_new) 
    new_model.matfac.X .= 0 
    new_model.sample_ids = sample_ids
    new_model.sample_conditions = sample_conditions
    
    # If batch info is provided, construct a 
    # new BatchShift layer and BatchArrayReg.
    if batch_dict != nothing
        new_layer = BatchShift(feature_views, batch_dict)
        new_model.matfac.col_trans.layers = (new_model.matfac.col_trans.layers[1:end-1]...,
                                             new_layer)

        # The new regularizer should be informed by the values of the
        # fitted model's batch shift.
        new_reg = construct_new_batch_reg!(new_layer.theta,
                                           model.matfac.col_trans_reg.regs[end],
                                           model.matfac.col_transform.layers[end].theta)
        new_model.matfac.col_trans_reg.regs = (new_model.matfac.col_trans_reg.regs[1:end-1]...,
                                               new_reg)
    end

    # If sample conditions are provided,
    # construct a new GroupRegularizer for sample conditions
    if sample_conditions != nothing
        new_model.matfac.X_reg.regularizers = (new_model.matfac.X_reg.regularizers[1],
                                               construct_new_group_reg(sample_conditions,
                                                                       model.matfac.X_reg.regularizers[2],
                                                                       model.matfac.X),
                                               new_model.matfac.X_reg.regularizers[3])
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
        mf_fit!(new_model; opt=opt, verbosity=verbosity-1,
                           max_epochs=500, update_col_layers=true)
    end

    # Update the new X 
    mf_fit!(new_model; update_X=true, verbosity=verbosity,
                       opt=opt, fit_kwargs...)

    # Finally: update them both, jointly 
    mf_fit!(new_model; update_X=true, update_col_layers=true,
                       verbosity=verbosity, opt=opt, fit_kwargs...)

    new_model = cpu(new_model)
    unfreeze_layer!(new_model.matfac.col_transform, 1:3)

    #############################################################
    # Restore the model as it was, and assemble the return values
    #############################################################

    # Restore the original data to the original model
    model.data = old_data

    return new_model 
end



