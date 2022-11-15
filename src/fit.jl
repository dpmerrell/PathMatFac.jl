
import StatsBase: fit!


##########################
# Helper functions

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
    
    verbose_print("Computing λ_Y max...\n"; verbosity=verbosity, level=1)

    M, N = size(D) 
    K, N = size(model.matfac.Y)

    # Curry the data loss to be a function of Y and the data 
    Y_loss_fn = (m, Y, D) -> MF.data_loss(m.X, Y, 
                                          m.row_transform, m.col_transform,
                                          m.noise_model, D; calibrate=true)
                                                 
    # Set all entries of X to 1
    X_temp = copy(model.matfac.X)
    
    # Compute the full gradient of the data-loss w.r.t. Y, at Y=0 and extreme values of X.
    Y_zero = zero(model.matfac.Y)
    model.matfac.X .= 1 
    Y_zero_grad_p = MF.batched_reduce((g, m, D) -> g .+ gradient(Y -> Y_loss_fn(m, Y, D), Y_zero)[1], 
                                      model.matfac, D; capacity=capacity, start=zero(model.matfac.Y))
    model.matfac.X .= -1
    Y_zero_grad_n = MF.batched_reduce((g, m, D) -> g .+ gradient(Y -> Y_loss_fn(m, Y, D), Y_zero)[1], 
                                      model.matfac, D; capacity=capacity, start=zero(model.matfac.Y))
    abs_grads = map((g1,g2) -> max(g1,g2), abs.(Y_zero_grad_n), abs.(Y_zero_grad_p))
    max_grad = maximum(abs_grads)

    # The size of lambda_max is governed by the largest entry of the gradient: 
    lambda_max = max_grad * (K/M) * 4 # (include a "safety factor" of 4)
    verbose_print("λ_Y max = ", lambda_max, "\n"; verbosity=verbosity, level=1)

    # Restore the entries of X
    model.matfac.X .= X_temp

    return lambda_max 
end


function scale_column_losses!(model, D; capacity=Int(25e6), verbosity=1)
    verbose_print("Re-weighting column losses\n"; verbosity=verbosity)
    col_errors = MF.batched_column_mean_loss(model.matfac.noise_model, D; 
                                             capacity=capacity)
    weights = abs.(1 ./ col_errors)
    weights[ (!isfinite).(weights) ] .= 1
    MF.set_weight!(model.matfac.noise_model, weights)

    col_losses = MF.batched_column_mean_loss(model.matfac.noise_model, D)
end


function initialize_params!(model::MultiomicModel, D::AbstractMatrix;
                            capacity=Int(25e6), verbosity=1,
                            loss_map=DEFAULT_ASSAY_LOSSES,
                            mean_init_map=MEAN_INIT_MAP,
                            var_init_map=VAR_INIT_MAP)

    verbose_print("Setting initial values for mu and sigma...\n"; verbosity=verbosity, level=1)
    
    model_losses = map(x->loss_map[x], model.data_assays[model.used_feature_idx])
    unq_losses = unique(model_losses)
    mean_vec, var_vec = MF.batched_column_meanvar(D; capacity=capacity)

    # Transform the means and variances appropriately for each assay
    for ul in unq_losses
        ul_idx = model_losses .== ul
        mean_vec[ul_idx] .= (mean_init_map[ul]).(mean_vec[ul_idx], var_vec[ul_idx])
        var_vec[ul_idx] .= (var_init_map[ul]).(mean_vec[ul_idx], var_vec[ul_idx])
    end

    # Remove NaNs from the variance and mean vectors
    nan_idx = (!isfinite).(var_vec)
    MF.toone!(var_vec, nan_idx)
    nan_idx = (!isfinite).(mean_vec)
    MF.tozero!(mean_vec, nan_idx)

    # Initialize values of mu, logsigma
    model.matfac.col_transform.cshift.mu .= mean_vec 
    map!(x->max(x,1e-3), var_vec, var_vec)
    logsigma = log.(sqrt.(var_vec))
    model.matfac.col_transform.cscale.logsigma .= logsigma 

    ## Initialize values of X and Y in such a way that they
    ## are consistent with pathway priors
    model.matfac.Y[model.matfac.Y_reg.l1_reg.l1_idx] .= 0
end


"""
    fit!(model::MultiomicModel, D::AbstractMatrix;
                                fit_hyperparam=true,
                                lambda_max=nothing,
                                n_lambda=8,
                                lambda_min_ratio=1e-3,
                                capacity=25e6,
                                verbosity=1, kwargs...)

    Fit a model to data. By default, tune the regularizer weights.
    The regularizer weights are varied in a way that preserves
    the relative sizes of those specified in `model`.
"""
function fit!(model::MultiomicModel, D::AbstractMatrix; 
              fit_hyperparam=true, capacity=Int(25e6), 
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


function postprocess!(fitted_model)

    # Remove sign ambiguity from factorization:
    # choose the sign that maximizes the number
    # of pathway members with positive Y-components.
    non_pwy_idx = fitted_model.matfac.Y_reg.l1_reg.l1_idx
    K = size(fitted_model.matfac.X,1)

    for k=1:K
        pwy_idx = (!).(non_pwy_idx[k,:])

        if dot(pwy_idx, sign.(fitted_model.matfac.Y[k,:])) < 0
            fitted_model.matfac.Y[k,:] .*= -1
            fitted_model.matfac.X[k,:] .*= -1
        end
    end
end


function fit_fixed_weight!(model::MultiomicModel, D::AbstractMatrix; kwargs...)

    # train the matrix factorization model
    MF.fit!(model.matfac, D; kwargs...)
    return model
end


function fit_reg_path!(model::MultiomicModel, D::AbstractMatrix; verbosity=1,
                       lambda_max=nothing, lambda_min_ratio=1e-3,
                       n_lambda=8, update_criterion=precision_selection, 
                       inner_callback_type=MatFac.HistoryCallback,
                       outer_callback=nothing, capacity=Int(25e6),
                       kwargs...)
    # Reweight the column losses.
    # Need to do this _before_ we choose lambda_max
    scale_column_losses!(model, D; capacity=capacity, verbosity=verbosity)

    # Set default parameter values
    if lambda_max == nothing
        lambda_max = select_lambda_max(model, D; capacity=capacity, verbosity=verbosity) 
    end
    lambda_min = lambda_max * lambda_min_ratio
    if outer_callback == nothing
        outer_callback = OuterCallback()
    end

    # Initialize the latent factors to have very small entries
    # (this is appropriate for regularizer-path hyperparameter selection)
    model.matfac.X .*= 1e-5
    model.matfac.Y .*= 1e-5
        
    # Initialize some loop variables
    log_lambdas = collect(range(log(lambda_max), 
                                log(lambda_min); 
                                length=n_lambda))
    lambdas = exp.(log_lambdas)

    # Store the specified regularizer weights
    lambda_Y = model.matfac.lambda_Y 
    lambda_X = model.matfac.lambda_X
    lambda_layers = model.matfac.lambda_col
    # ...we will preserve their relative sizes.
    lambda_X /= lambda_Y
    lambda_layers /= lambda_Y
    lambda_Y = 1.0

    # Keep track of the best model we've seen thus far.
    best_model = nothing

    # Loop through values of lambda
    for iter=1:n_lambda

        this_lambda = lambdas[iter]
        # Set the regularizer weights
        model.matfac.lambda_Y = this_lambda
        model.matfac.lambda_X = this_lambda * lambda_X
        model.matfac.lambda_col = this_lambda * lambda_layers
 
        # initialize an "inner" callback 
        inner_callback = inner_callback_type()

        # Outer loop printout
        verbose_print("Outer iteration ", iter, "; λ_Y = ", 
                      round(lambdas[iter]; digits=5), "\n"; 
                      verbosity=verbosity, level=1)
        fit_fixed_weight!(model, D; callback=inner_callback,
                                    scale_column_losses=false, 
                                    verbosity=verbosity, 
                                    capacity=capacity, kwargs...)
        
        # Call the outer callback for this iteration
        outer_callback(model, inner_callback) 

        # Check whether to update the returned model
        if (iter == 1) || update_criterion(model, best_model, D, iter; capacity=capacity)
            best_model = deepcopy(model)
        end

    end

    model.matfac = best_model.matfac
    return model
end


