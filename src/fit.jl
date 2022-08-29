
import ScikitLearnBase: fit!


function initialize_params!(model::MultiomicModel, D::AbstractMatrix;
                            capacity=Int(25e6), verbose=true,
                            loss_map=DEFAULT_ASSAY_LOSSES,
                            mean_init_map=MEAN_INIT_MAP,
                            var_init_map=VAR_INIT_MAP)

    if verbose
        println("Setting initial values for mu and sigma...")
    end
    
    M, N = size(D)
    batch_size = Int(round(capacity / N))
    model_losses = map(x->loss_map[x], model.data_assays[model.used_feature_idx])
    unq_losses = unique(model_losses)
    mean_vec, var_vec = MF.column_meanvar(D, batch_size)

    # Transform the means and variances appropriately for each assay
    for ul in unq_losses
        ul_idx = model_losses .== ul
        mean_vec[ul_idx] .= (mean_init_map[ul]).(mean_vec[ul_idx], var_vec[ul_idx])
        var_vec[ul_idx] .= (var_init_map[ul]).(mean_vec[ul_idx], var_vec[ul_idx])
    end
    # Remove NaNs
    nan_idx = isnan.(var_vec)
    MF.tozero!(var_vec, nan_idx)
    nan_idx = isnan.(mean_vec)
    MF.tozero!(mean_vec, nan_idx)

    # Initialize values of mu, logsigma
    model.matfac.col_transform.cshift.mu .= mean_vec 
    map!(x->max(x,1e-3), var_vec, var_vec)
    logsigma = log.(sqrt.(var_vec))
    model.matfac.col_transform.cscale.logsigma .= logsigma 

end


function postprocess!(fitted_model)

    # Remove sign ambiguity from factorization:
    # choose the sign that maximizes the number
    # of pathway members with positive Y-components.
    non_pwy_idx = fitted_model.matfac.Y_reg.l1_feat_idx
    K = size(fitted_model.matfac.X,1)

    for k=1:K
        pwy_idx = (!).(non_pwy_idx[k])

        if dot(pwy_idx, sign.(fitted_model.matfac.Y[k,:])) < 0
            fitted_model.matfac.Y[k,:] .*= -1
            fitted_model.matfac.X[k,:] .*= -1
        end
    end

end


function fit!(model::MultiomicModel, D::AbstractMatrix; fit_hyperparam=false,
                                                        capacity=Int(25e6), kwargs...)
    # Permute the data columns to match the model's
    # internal ordering
    println("Rearranging data columns...")
    D_r = D[:, model.used_feature_idx]

    # If on GPU, de-allocate the original data
    if typeof(D) <: CuArray
        CUDA.unsafe_free!(D)
    end
   
    # Initialize the parameters     
    initialize_params!(model, D_r; capacity=capacity)

    # Perform the actual fit
    if fit_hyperparam
        fit_reg_path!(model, D_r; capacity=capacity, kwargs...)
    else
        # Set some model parameters to the right ball-park
        fit_fixed_weight!(model, D_r; capacity=capacity, kwargs...)
    end
    
    # de-allocate the rearranged data now
    if typeof(D_r) <: CuArray
        CUDA.unsafe_free!(D_r)
    else
        D_r = nothing
    end

    # Postprocess the model 
    postprocess!(model)

    return model
end


function fit_fixed_weight!(model::MultiomicModel, D::AbstractMatrix; kwargs...)

    # train the matrix factorization model
    fit!(model.matfac, D; kwargs...)
    return model
end


function verbose_print(args...; verbosity=1, level=1)
    if verbosity >= level
        print(string(args...))
    end
end


function fit_reg_path!(model::MultiomicModel, D::AbstractMatrix; verbosity=1,
                                                                 init_lambda_Y=128.0,
                                                                 shrink_factor=0.5, 
                                                                 update_criterion=latest_model, 
                                                                 term_condition=iter_termination,
                                                                 callback=MatFac.HistoryCallback,
                                                                 outer_callback=OuterCallback,
                                                                 history_json="histories.json",
                                                                 kwargs...)

    # Initialize the latent factors to have very small entries
    # (this is appropriate for regularizer-path hyperparameter selection)
    model.matfac.X .*= 1e-5
    model.matfac.Y .*= 1e-5
        
    # Initialize some loop variables
    lambda = init_lambda_Y
    iter = 1
    macro_callback = outer_callback()
    macro_callback.history_json = history_json

    # Keep track of the best model we've seen thus far
    best_model = deepcopy(cpu(model))

    # Loop through values of lambda_Y
    while true
        # Set the regularizer weight
        model.matfac.lambda_Y = lambda
        
        # initialize an "inner" callback 
        micro_callback = callback()

        # Fit the model
        verbose_print("Outer iteration ", iter, "; λ_Y = ", lambda, "\n"; verbosity=verbosity, level=1)
        fit_fixed_weight!(model, D; callback=micro_callback, 
                                    verbosity=verbosity, kwargs...)

        # Call the outer callback for this iteration
        macro_callback(model, micro_callback) 

        # Check whether to update the returned model
        if update_criterion(model, best_model, D, iter)
            best_model = deepcopy(cpu(model))
        end 

        # Check termination condition
        if term_condition(model, best_model, D, iter)
            break
        else
            iter += 1
            lambda *= shrink_factor
        end

    end

    model.matfac = best_model.matfac

    return model
end


