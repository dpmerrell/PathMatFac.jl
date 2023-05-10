

export transform


function transform(model::PathMatFacModel, D::AbstractMatrix;
                   use_gpu::Bool=true,
                   feature_ids::Union{<:AbstractVector,Nothing}=nothing,
                   sample_ids::Union{<:AbstractVector,Nothing}=nothing,
                   feature_views::Union{<:AbstractVector,Nothing}=nothing,
                   verbosity=1, print_prefix="", 
                   max_epochs=1000, lr=1.0, fit_kwargs...)

    K, N = size(model.matfac.Y)
    M_new, N_new = size(D)

    next_pref = string("    ", print_prefix)
    v_println("Transforming new data..."; verbosity=verbosity, prefix=print_prefix)

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

    if sample_ids != nothing
        @assert length(sample_ids) == M_new "`sample_ids` must have length == size(D,1)"
    else
        sample_ids = collect(1:M_new)
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
    set_layer!(new_model.matfac.col_transform, 1, view(new_model.matfac.col_transform.layers[1], :, old_idx))
    set_layer!(new_model.matfac.col_transform, 3, view(new_model.matfac.col_transform.layers[3], :, old_idx))
    new_model.matfac.Y_reg = x->0.0
    new_model.matfac.noise_model = model.matfac.noise_model[old_idx]
    new_model.feature_views = model.feature_views[old_idx]
    new_model.feature_ids = feature_ids[new_idx]

    # Construct "sample attributes"
    new_model.matfac.X = similar(model.matfac.X, K, M_new) 
    new_model.matfac.X .= 0
    new_model.matfac.X_reg = x->0.0 
    new_model.sample_ids = sample_ids
   
    ##################################################
    # Fit the model to the new data.
    ##################################################
    
    # Move to GPU; initialize an optimizer
    if use_gpu
        new_model = gpu(new_model)
    end

    # Freeze the other layers and ignore the column
    # parameters' regularizers
    freeze_layer!(new_model.matfac.col_transform, 1:4)
    freeze_reg!(new_model.matfac.col_transform_reg, 1:4)

    # Update the new X 
    mf_fit_adapt_lr!(new_model; update_X=true,  
                                verbosity=verbosity, print_prefix=next_pref,
                                max_epochs=max_epochs, lr=lr, fit_kwargs...)

    if use_gpu
        new_model = cpu(new_model)
    end

    unfreeze_layer!(new_model.matfac.col_transform, 1:4)

    #############################################################
    # Restore the model as it was, and assemble the return values
    #############################################################

    # Restore the original data to the original model
    model.data = old_data

    return new_model 
end



