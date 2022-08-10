
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


function fit_fixed_weight!(model::MultiomicModel, D::AbstractMatrix; capacity=Int(25e6), kwargs...)

    # Set some model parameters to the right ball-park
    initialize_params!(model, D; capacity=capacity)

    # train the matrix factorization model
    fit!(model.matfac, D; capacity=capacity, kwargs...)

    return model
end


function fit_reg_path!(model::MultiomicModel, D::AbstractMatrix; capacity=Int(25e6), 
                                                                 init_lambda_Y=10.0, 
                                                                 term_condition=nothing, 
                                                                 histories_json="histories.json",
                                                                 kwargs...)
    # If no termination condition is provided, then
    # just run 10 iterations. This can be useful for
    # exploration.
    if term_condition == nothing
        term_condition = (model, iter) -> iter >= 10 
    end

    # Initialize some loop variables
    lambda = init_lambda_Y
    iter = 1
    histories = []

    while true
        # Set the regularizer weight
        model.matfac.Y_reg.l1_weight = lambda
        model.matfac.Y_reg.net_weight = lambda
        
        # Store the history for each call of `fit`
        history_callback = MatFac.HistoryCallback()

        # Fit the model
        fit_fixed_weight!(model, D; capacity=capacity, callback=history_callback, kwargs...)
        push!(histories, history_callback.history)

        # Save histories to a JSON file
        open(histories_json, "w") do f
            JSON.print(f, histories)
        end

        # Check termination condition
        if term_condition(model, iter)
            break
        else
            iter += 1
            lambda *= 0.5
        end

    end

    return model
end


