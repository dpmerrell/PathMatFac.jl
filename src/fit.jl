
import ScikitLearnBase: fit!


function verbose_print(args...; verbosity=1, level=1)
    if verbosity >= level
        print(string(args...))
    end
end

function randn_like(A::AbstractMatrix)
    M, N = size(A)
    if typeof(A) <: CuArray
        return CUDA.randn(M,N)
    end
    randn(M,N)
end

function select_lambda_max(model::MultiomicModel, D::AbstractMatrix;
                           capacity=Int(25e6), verbosity=1)
    
    verbose_print("Setting λ_Y max...\n"; verbosity=verbosity, level=1)

    M, N = size(D) 
    row_batch_size = div(capacity,N)
    K, N = size(model.matfac.Y)
    Y_zero_grad = zero(model.matfac.Y)
    Y_zero = zero(model.matfac.Y)

    test_X = randn_like(model.matfac.X)

    # Compute the full gradient of the data-loss w.r.t. Y at Y=0 and random X.
    for row_batch in MF.BatchIter(M, row_batch_size)

        X_view = view(test_X, :, row_batch)
        row_trans_view = view(model.matfac.row_transform, row_batch, 1:N)
        col_trans_view = view(model.matfac.col_transform, row_batch, 1:N)
        D_view = view(D, row_batch, 1:N)

        data_loss_Y_fn = Y -> MF.data_loss(X_view, Y,
                                                row_trans_view,
                                                col_trans_view,
                                                model.matfac.noise_model,
                                           D_view)

        (g,) = gradient(data_loss_Y_fn, Y_zero)

        Y_zero_grad .+= g 
    end    

    # The size of lambda_max is governed by the largest entry of the gradient:
    lambda_max = maximum(abs.(Y_zero_grad)) * (K/M)
    verbose_print("λ_Y max = ", lambda_max, "\n"; verbosity=verbosity, level=1)

    return lambda_max 
end

function initialize_params!(model::MultiomicModel, D::AbstractMatrix;
                            capacity=Int(25e6), verbosity=1,
                            loss_map=DEFAULT_ASSAY_LOSSES,
                            mean_init_map=MEAN_INIT_MAP,
                            var_init_map=VAR_INIT_MAP)

   verbose_print("Setting initial values for mu and sigma...\n"; verbosity=verbosity, level=1)
    
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

    # Remove NaNs from the variance and mean vectors
    nan_idx = isnan.(var_vec)
    MF.tozero!(var_vec, nan_idx)
    nan_idx = isnan.(mean_vec)
    MF.tozero!(mean_vec, nan_idx)

    # Initialize values of mu, logsigma
    model.matfac.col_transform.cshift.mu .= mean_vec 
    map!(x->max(x,1e-3), var_vec, var_vec)
    logsigma = log.(sqrt.(var_vec))
    model.matfac.col_transform.cscale.logsigma .= logsigma 

    # Initialize values of X and Y in such a way that they
    # are consistent with pathway priors
    for (k, idx_vec) in enumerate(model.matfac.Y_reg.l1_feat_idx)
        model.matfac.Y[k,idx_vec] .= 0
    end

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


function fit!(model::MultiomicModel, D::AbstractMatrix; fit_hyperparam=true,
                                                        capacity=Int(25e6), 
                                                        verbosity=1, kwargs...)
    # Permute the data columns to match the model's
    # internal ordering
    verbose_print("Rearranging data columns...\n"; verbosity=verbosity)

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




function fit_reg_path!(model::MultiomicModel, D::AbstractMatrix; verbosity=1,
                                                                 init_lambda_Y=nothing,
                                                                 shrink_factor=0.5, 
                                                                 update_criterion=latest_model, 
                                                                 term_condition=iter_termination,
                                                                 callback=MatFac.HistoryCallback,
                                                                 outer_callback=OuterCallback,
                                                                 history_json="histories.json",
                                                                 capacity=Int(25e6),
                                                                 kwargs...)
    # Default parameter values
    if init_lambda_Y == nothing
        init_lambda_Y = select_lambda_max(model, D; capacity=capacity, verbosity=verbosity) 
    end

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
        reweight_columns = (iter == 1)
        fit_fixed_weight!(model, D; callback=micro_callback,
                                    scale_column_losses=reweight_columns, 
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


