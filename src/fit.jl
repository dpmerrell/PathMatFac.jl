
import StatsBase: fit!


######################################################################
# Wrapper for MatFac.fit!
######################################################################

function mf_fit!(model::PathMatFacModel; scale_column_losses=false,
                                         update_X=false,
                                         update_Y=false,
                                         update_row_layers=false,
                                         update_col_layers=false,
                                         update_noise_models=true,
                                         reg_relative_weighting=false,
                                         update_X_reg=false,
                                         update_Y_reg=false,
                                         update_row_layers_reg=false,
                                         update_col_layers_reg=false,
                                         keep_history=true,
                                         kwargs...)

    # Change some of the default kwarg values
    h = MF.fit!(model.matfac, model.data; scale_column_losses=scale_column_losses, 
                                          update_X=update_X,
                                          update_Y=update_Y,
                                          update_row_layers=update_row_layers,
                                          update_col_layers=update_col_layers,
                                          update_noise_models=update_noise_models,
                                          update_X_reg=update_X_reg,
                                          update_Y_reg=update_Y_reg,
                                          update_row_layers_reg=update_row_layers_reg,
                                          update_col_layers_reg=update_col_layers_reg,
                                          keep_history=keep_history,
                                          t_start=FIT_START_TIME,
                                          kwargs...)
    return h
end


function construct_optimizer(model, lr)
    return Flux.Optimise.AdaGrad(lr)
end


function mf_fit_adapt_lr!(model::PathMatFacModel; lr=1.0,
                                                  min_lr=0.001, 
                                                  max_epochs=1000,
                                                  history=nothing,
                                                  keep_history=true,
                                                  verbosity=1,
                                                  print_prefix="", 
                                                  kwargs...)
    
    opt = construct_optimizer(model, lr)
    epoch = 1
    while epoch <= max_epochs
        h = mf_fit!(model; opt=opt, max_epochs=max_epochs, epoch=epoch, keep_history=true, 
                           print_prefix=print_prefix, verbosity=verbosity, 
                           kwargs...)
        history!(history, h; name=string("mf_fit_lr=", opt.eta))
        
        if h["term_code"] == "loss_increase"
            opt.eta *= Float32(0.5)
            if opt.eta < min_lr
                break
            end
            v_println("Resuming with smaller learning rate (", opt.eta, ")"; verbosity=verbosity, prefix=print_prefix)
            epoch = h["epochs"]
        else
            break
        end
    end

end


########################################################################
# PARAMETER INITIALIZATION
########################################################################

function init_mu!(model::PathMatFacModel; capacity=Int(10e8), lr_mu=0.1, max_epochs=500, 
                                          verbosity=1, print_prefix="", history=nothing, kwargs...)
    
    keep_history = (history != nothing)

    opt = construct_optimizer(model, lr_mu)
    result = MF.compute_M_estimates(model.matfac, model.data;
                                    capacity=capacity, max_epochs=max_epochs,
                                    opt=opt, verbosity=verbosity, print_prefix=print_prefix,
                                    keep_history=keep_history,
                                    rel_tol=1e-5, abs_tol=1e-3) 
    # Awkwardly handle the variable-length output
    h = nothing
    M_estimates = result
    if keep_history
        M_estimates = result[1]
        h = result[2]
        history!(history, h; name="init_mu")
    end

    model.matfac.col_transform.layers[3].mu .= vec(M_estimates)
end


function init_theta!(model::PathMatFacModel; capacity=Int(10e8), max_epochs=500,
                                             lr_theta=1.0,
                                             verbosity=1, print_prefix="", 
                                             history=nothing, kwargs...)
    # Freeze everything except batch shift... 
    freeze_layer!(model.matfac.col_transform, 1:3)

    mf_fit_adapt_lr!(model; lr=lr_theta, 
                     update_col_layers=true, capacity=capacity,
                     max_epochs=max_epochs, 
                     verbosity=verbosity,
                     print_prefix=print_prefix,
                     history=history)

    unfreeze_layer!(model.matfac.col_transform, 1:3)
    history!(history; name="init_theta") 
end


function init_logsigma!(model::PathMatFacModel; capacity=Int(10e8), history=nothing)

    # *IMPORTANT*
    # We assume the model's column shift has
    # already been set.

    # While computing the `link_col_sqerr`, we need to temporarily set
    # the matrix factorization's X and Y to zero  
    orig_X = model.matfac.X
    orig_Y = model.matfac.Y
    model.matfac.X = zero(orig_X)
    model.matfac.Y = zero(orig_Y)
 
    col_vars = MF.link_col_sqerr(model.matfac.noise_model, model.matfac, model.data; 
                                 capacity=capacity)
    col_vars ./= vec(MF.column_nonnan(model.data))
    
    K = size(model.matfac.X, 1)
    model.matfac.col_transform.layers[1].logsigma .= log.(sqrt.(col_vars))# ./ sqrt(K))

    model.matfac.X = orig_X
    model.matfac.Y = orig_Y
    history!(history; name="init_logsigma") 
end


function reweight_col_losses!(model::PathMatFacModel; capacity=Int(10e8), history=nothing)
    
    # Set the noise model's weights to 1
    M, N = size(model.data)
    one_vec = similar(model.data,N)
    one_vec .= 1
    MF.set_weight!(model.matfac.noise_model, one_vec)

    # Temporarily set X and Y to zero
    orig_X = model.matfac.X
    orig_Y = model.matfac.Y
    model.matfac.X = zero(orig_X)
    model.matfac.Y = zero(orig_Y)

    # Compute the RMS of the loss gradient for each column
    ssq_grads = vec(MF.batched_column_ssq_grads(model.matfac,
                                                model.data; 
                                                capacity=capacity)
                   )
    rms_grads = sqrt.(ssq_grads ./ M) # NOTE: divide by M (rather than the number of non-NaN entries.)
                                      #       This increases the weight of columns containing
                                      #       many NaNs, compensating for their "lack of gradient."

    # The RMS gradients should be multiplied by the column scales
    rms_grads .*= exp.(model.matfac.col_transform.layers[1].logsigma)
    weights = 1 ./ rms_grads 
    weights[ (!isfinite).(weights) ] .= 1
    
    # Set the weights
    MF.set_weight!(model.matfac.noise_model, vec(weights))

    # Restore X and Y
    model.matfac.X = orig_X
    model.matfac.Y = orig_Y

    history!(history; name="reweight_col_losses")
end


function rec_set_thresholds(p_vec, l_p::Float32, r_p::Float32)

    if length(p_vec) == 0
        # If zero probabilities remain, then we're done 
        # (we had an even number of probabilities)
        return Float32[] 
    elseif length(p_vec) == 1
        # If one probability remains, then we're done 
        # (we had an odd number of probabilities)
        return Float32[] 
    else
        # If more probabilities remain, then we 
        # remove the two "outermost" probabilities from p_vec;
        # add those probabilities to l_p, r_p;
        l_p += p_vec[1]
        r_p += p_vec[end] 
        p_vec = p_vec[2:1-end]

        # compute thresholds for them;
        new_l_th = inv_logistic(l_p)
        new_r_th = inv_logistic(1 - r_p)
 
        # call rec_set_thresholds on the remaining probabilities;
        # and concatenate the results together 
        return vcat([new_l_th], 
                    rec_set_thresholds(p_vec, l_p, r_p),
                    [new_r_th])
    end
    
end


function init_ordinal_thresholds!(model::PathMatFacModel; verbosity=1, print_prefix="")

    nm = model.matfac.noise_model
    data = model.data

    for (n, cr) in zip(nm.noises, nm.col_ranges)
        if isa(n, MF.OrdinalNoise)
            v_println("Setting ordinal thresholds..."; verbosity=verbosity,
                                                       prefix=print_prefix)
            K = length(n.ext_thresholds)
            p = zeros(Float32, K-1)
            
            ord_data = view(data, :, cr)
            for k=1:(K-1)
                k_idx = (ord_data .== Float32(k))
                p[k] = sum(k_idx) + 1
            end
            p ./= sum(p)

            new_thresh = rec_set_thresholds(p, Float32(0.0), Float32(0.0))
            n.ext_thresholds[2:(end-1)] .= new_thresh

        end
    end
end


function init_factors!(model::PathMatFacModel; verbosity=1, 
                                               print_prefix="", 
                                               history=nothing,
                                               lr=1.0,
                                               capacity=10^8,
                                               max_epochs=1000,
                                               init_factors_method="adagrad",
                                               rel_tol=1e-5,
                                               abs_tol=1e-5,
                                               backtrack_shrinkage=0.8,
                                               kwargs...)
    n_prefix = string(print_prefix, "    ")
    if init_factors_method == "lbfgs"
        v_println("Initializing linear factors X,Y via L-BFGS..."; verbosity=verbosity,
                                                                   prefix=print_prefix)
        orig_X_reg = model.matfac.X_reg
        orig_Y_reg = model.matfac.Y_reg
        model.matfac.X_reg = x->Float32(0.05).*sum(x.*x)
        model.matfac.Y_reg = y->Float32(0.05).*sum(y.*y)
        fit_lbfgs!(model.matfac, model.data; verbosity=verbosity,
                                             print_prefix=n_prefix,
                                             max_iter=max_epochs,
                                             rel_tol=rel_tol,
                                             abs_tol=abs_tol,
                                             backtrack_shrinkage=backtrack_shrinkage)
        model.matfac.X_reg = orig_X_reg
        model.matfac.Y_reg = orig_Y_reg
        history!(history; name="init_factors_lbfgs")
    else
        v_println("Initializing linear factors X,Y via AdaGrad..."; verbosity=verbosity,
                                                                    prefix=print_prefix)
        mf_fit_adapt_lr!(model; capacity=capacity, update_X=true, update_Y=true,
                                lr=lr, min_lr=0.05, max_epochs=max_epochs, 
                                verbosity=verbosity, print_prefix=n_prefix,
                                history=history,
                                kwargs...)
        history!(history; name="init_factors")
    end

end


########################################################################
# BATCH EFFECT MODELING
########################################################################

# Method of Moments estimators for setting the
# theta and delta priors
function theta_mom(theta_values::Tuple)
    theta_mean = map(v -> mean(v, dims=2), theta_values)
    theta_var = map(v -> var(v, dims=2), theta_values)
    return theta_mean, theta_var
end

function delta2_mom(delta2_values::Tuple)

    delta2_mean = map(v -> mean(v, dims=2), delta2_values)
    delta2_var = map(v -> var(v, dims=2), delta2_values)

    alpha = map((m,v) -> Float32(2) .+ (m.*m)./(v .+ Float32(1e-9)), delta2_mean, delta2_var)
    beta = map((m,a) -> m .* (a .- Float32(1)), delta2_mean, alpha)
    return alpha, beta
end

function nans_to_val!(tup, val)
    nan_idx = map(t->(!isfinite).(t), tup)
    for (t,i) in zip(tup, nan_idx)
        t[i] .= val
    end
end

function assign_values!(tup1, tup2)
    for (t1,t2) in zip(tup1, tup2)
        t1 .= t2
    end
end

function theta_delta_em(model::MF.MatFacModel, delta2::Tuple, sigma2::AbstractVector, data::AbstractMatrix; 
                        capacity=10^8, batch_em_max_iter=100, batch_em_rtol=1e-6, verbosity=1, print_prefix="", history=nothing)

    theta = model.col_transform.layers[4].theta
    theta_lsq = deepcopy(theta.values)
    theta_old = deepcopy(theta.values)
    batch_sizes = ba_map(d->isfinite.(d), theta, data)
        
    diffs = MutableLinkedList() 
    for iter=1:batch_em_max_iter
    
        # Update priors  
        theta_mean, theta_var = theta_mom(theta.values)
        alpha, beta = delta2_mom(delta2)

        # Update batch shift (theta)
        assign_values!(theta_old, theta.values)
        theta.values = map((e_th, v_th, d2, th_lsq, bs, cr)-> (e_th.*d2.*transpose(view(sigma2, cr)) .+ th_lsq.*bs.*v_th)./(transpose(view(sigma2, cr)).*d2 .+ bs.*v_th), 
                           theta_mean, theta_var, delta2, theta_lsq, batch_sizes, theta.col_ranges)
        nans_to_val!(theta.values, Float32(0))
 
        # Update batch scale (delta^2)
        batch_col_sqerr = ba_map((m, d) -> MF.sqerr_func(m,d),
                                 theta, model, data)
        nans_to_val!(batch_col_sqerr, Float32(0))
 
        delta2 = map((a, b, sq, bs, cr) -> (b .+ Float32(0.5).*(sq ./ transpose(view(sigma2, cr)))) ./ (a .+ (Float32(0.5).*bs) .- Float32(1)), 
                     alpha, beta, batch_col_sqerr, batch_sizes, theta.col_ranges)
        nans_to_val!(delta2, 1) 

        theta_diff = sum(map((th,th_old)->sum((th .- th_old).^2), theta.values, theta_old))/sum(map(th->sum(th.*th), theta.values))
        v_println("(",iter, ") ||θ - θ'||^2/||θ||^2 : ", theta_diff; verbosity=verbosity, prefix=print_prefix)
        push!(diffs, theta_diff)

        if theta_diff < batch_em_rtol
            break
        end 
    end

    history!(history; name="batch_effect_EM_procedure", diffs=collect(diffs))

    return theta.values, delta2
end


function init_batch_effects!(model::PathMatFacModel; capacity=10^8, 
                                                     max_epochs=5000,
                                                     lr_regress=0.25,
                                                     lr_mu=0.1,
                                                     batch_em=true,
                                                     batch_em_rtol=1e-6,
                                                     batch_em_max_iter=100,
                                                     verbosity=1, print_prefix="",
                                                     history=nothing, kwargs...)

    n_pref = string(print_prefix, "    ")

    # Set up a matrix factorization model with the sole
    # purpose of initializing batch effects. 
    orig_matfac = model.matfac
    model.matfac = deepcopy(orig_matfac)

    # Its X matrix will store sample conditions;
    # its Y matrix will regress each column against them.
    condition_mat = ids_to_ind_mat(model.sample_conditions)
    M, K_conditions = size(condition_mat)
    N = size(model.matfac.Y, 2)
    model.matfac.X = similar(model.matfac.X, K_conditions, M)
    set_array!(model.matfac.X, transpose(condition_mat))
    model.matfac.Y = zeros_like(model.matfac.Y, K_conditions, N)

    v_println("Fitting column shifts..."; verbosity=verbosity,
                                          prefix=print_prefix)
    # Fit its column shifts
    init_mu!(model; capacity=capacity, max_epochs=max_epochs, lr_mu=lr_mu, 
                    verbosity=verbosity-1, print_prefix=n_pref,
                    history=history, kwargs...)
    orig_matfac.col_transform.layers[3].mu .= model.matfac.col_transform.layers[3].mu

    v_println("Regressing against sample conditions..."; verbosity=verbosity,
                                                         prefix=print_prefix)
    # Fit its Y factor (without regularization).
    model.matfac.Y_reg = y->0
    mf_fit_adapt_lr!(model; capacity=capacity, max_epochs=max_epochs,
                            lr=lr_regress,
                            min_lr=0.05,
                            verbosity=verbosity-1, print_prefix=n_pref,
                            scale_column_losses=false,
                            update_X=false,
                            update_Y=true,
                            update_col_layers=false,
                            update_X_reg=false,
                            update_Y_reg=false,
                            update_row_layers_reg=false,
                            update_col_layers_reg=false,
                            history=history)
    history!(history; name="regress_against_sample_conditions")

    # Fit its batch shift (without regularization; "least-squares")
    model.matfac.col_transform_reg = l->0
    v_println("Fitting batch shift..."; verbosity=verbosity,
                                        prefix=print_prefix)
    init_theta!(model; capacity=capacity, max_epochs=max_epochs,
                       verbosity=verbosity-1, print_prefix=n_pref, 
                       history=history, kwargs...)
    nans_to_val!(model.matfac.col_transform.layers[4].theta.values, Float32(0))
    theta_ba = deepcopy(model.matfac.col_transform.layers[4].theta)

    # Set its column scales
    v_println("Computing column scales..."; verbosity=verbosity,
                                            prefix=print_prefix)
    col_vars = vec(MF.link_col_sqerr(model.matfac.noise_model,
                                     model.matfac,
                                     model.data; capacity=capacity))
    col_vars ./= vec(MF.column_nonnan(model.data))
    # Set NaNs to 1
    col_vars[(!isfinite).(col_vars)] .= 1

    # Compute batchwise variances
    v_println("Computing batch scales..."; verbosity=verbosity,
                                           prefix=print_prefix)
    ba_sqerr = ba_map((m, d) -> MF.sqerr_func(m,d),
                      theta_ba, model.matfac, model.data)
    ba_M = ba_map(d -> isfinite.(d), theta_ba, model.data)
    ba_vars = map((sq, M) -> sq ./ M, ba_sqerr, ba_M)
    nans_to_val!(ba_vars, Float32(1)) # Set NaNs to 1

    # Divide batch vars by column vars to obtain "raw" delta^2's
    col_ranges = theta_ba.col_ranges
    delta2_values = map((v, cr) -> v ./ transpose(view(col_vars, cr)), ba_vars, col_ranges)

    # Update the estimated batch parameters in an EB fashion
    theta_values = model.matfac.col_transform.layers[4].theta.values
    if batch_em
        v_println("Batch effect EM procedure:"; prefix=print_prefix, verbosity=verbosity)
        theta_values, delta2_values = theta_delta_em(model.matfac, delta2_values, col_vars, model.data;
                                                     capacity=capacity,
                                                     batch_em_max_iter=batch_em_max_iter,
                                                     batch_em_rtol=batch_em_rtol,
                                                     print_prefix=n_pref, verbosity=verbosity)
    end

    # Set the model's parameters to the fitted values
    model.matfac = orig_matfac
    
    # We reduce the column scales by 1/√K so that the entries of X, Y take reasonable values
    K = size(model.matfac.X,1)
    model.matfac.col_transform.layers[1].logsigma = log.(sqrt.(col_vars))# ./ K)) 

    model.matfac.col_transform.layers[2].logdelta.values = map(v->log.(sqrt.(v)), delta2_values)
    model.matfac.col_transform.layers[4].theta.values = theta_values

end

#######################################################################
# POSTPROCESSING FUNCTIONS
#######################################################################

# "Whiten" the embedding X, such that std(X_k) = 1 for each k;
# Variance is reallocated to Y; and then reallocated to logsigma.
function whiten!(model::PathMatFacModel)
    X_std = std(model.matfac.X, dims=2)
    model.matfac.X ./= X_std
    model.matfac.Y .*= X_std

    # For each view, reallocate variance in Y to sigma.
    view_crs = ids_to_ranges(cpu(model.feature_views))
    for cr in view_crs
        # Compute row-wise standard deviations in Y:
        Y_view_std = std(view(model.matfac.Y, :, cr), dims=2)
        
        # Select the largest one and use it to 
        # rescale both Y and sigma.
        Y_std_max = maximum(Y_view_std)
        model.matfac.Y[:,cr] ./= Y_std_max

        model.matfac.col_transform.layers[1].logsigma[cr] .+= log(Y_std_max)
    end

end


function rotate_by_svd!(model::PathMatFacModel)

    # Compute SVD of Y
    F = svd(model.matfac.Y)
    U = F.U
    s = F.S
    Vt = F.Vt

    # Reassign Y <- S*Vt
    model.matfac.Y .= s .* Vt

    # Rotate Xt <- Xt * U
    model.matfac.X .= transpose(transpose(model.matfac.X)*U)
end


function reorder_by_importance!(model::PathMatFacModel)
    Y_ssq = cpu(vec(sum(model.matfac.Y .* model.matfac.Y, dims=2)))
    srt_idx = sortperm(Y_ssq, rev=true)

    model.matfac.X .= model.matfac.X[srt_idx,:]
    model.matfac.Y .= model.matfac.Y[srt_idx,:]
    
    reorder_reg!(model.matfac.Y_reg, srt_idx)
    reorder_reg!(model.matfac.X_reg, srt_idx)
end


#######################################################################
# FITTING PROCEDURES
#######################################################################

################################
# Procedures for simple models
function basic_fit!(model::PathMatFacModel; fit_batch=false, 
                                            fit_mu=false, fit_logsigma=false,
                                            reweight_losses=false,
                                            init_factors=false,
                                            init_factors_method="adagrad",
                                            fit_factors=false,
                                            init_ordinal=false,
                                            svd_rotate=false,
                                            whiten=false,
                                            capacity=Int(10e8),
                                            lr=1.0, max_epochs=1000,
                                            verbosity=1, print_prefix="",
                                            history=nothing,
                                            lr_regress=1.0,
                                            lr_mu=0.1,
                                            lr_theta=1.0,
                                            kwargs...)


    n_prefix = string(print_prefix, "    ")
    K,N = size(model.matfac.Y)
    keep_history = (history != nothing)
 
    if init_ordinal
        init_ordinal_thresholds!(model; verbosity=verbosity,
                                        print_prefix=print_prefix)
    end

    # Things differ pretty radically if we're fitting batch parameters
    # vs. only column parameters
    if fit_batch
        @assert isa(model.matfac.col_transform.layers[2], BatchScale) "Model must have batch parameters whenever `fit_batch` is true"
        @assert isa(model.matfac.col_transform.layers[4], BatchShift) "Model must have batch parameters whenever `fit_batch` is true"
        # Fit the batch shift parameters.
        v_println("Fitting batch parameters..."; verbosity=verbosity,
                                                 prefix=print_prefix)
        init_batch_effects!(model; capacity=capacity, 
                                   max_epochs=max_epochs,
                                   verbosity=verbosity,
                                   print_prefix=n_prefix,
                                   history=history,
                                   lr_regress=lr_regress,
                                   lr_theta=lr_theta)
    else
        if fit_mu
            # Initialize mu with the column M-estimates.
            v_println("Fitting column shifts..."; verbosity=verbosity,
                                                  prefix=print_prefix)
            init_mu!(model; capacity=capacity, 
                            max_epochs=500,
                            verbosity=verbosity,
                            print_prefix=n_prefix,
                            history=history)
        end
        # Choose column scales in a way that encourages
        # columns of Y to have similar magnitudes.
        if fit_logsigma
            v_println("Fitting column scales..."; verbosity=verbosity,
                                                  prefix=print_prefix)
            init_logsigma!(model; capacity=capacity)
        end
    end

    # Reweight the column losses so that they exert 
    # similar gradients on X
    if reweight_losses 
        v_println("Reweighting column losses..."; verbosity=verbosity,
                                                  prefix=print_prefix)
        reweight_col_losses!(model; capacity=capacity)
    end

    # Initialize the factors X,Y with some specialized algorithm.
    if init_factors
        init_factors!(model; lr=lr, init_factors_method=init_factors_method,
                             verbosity=verbosity, 
                             print_prefix=print_prefix, 
                             history=history,
                             capacity=capacity, 
                             max_epochs=max_epochs,
                             kwargs...)
    end

    # Fit the factors X,Y.
    if fit_factors
        v_println("Fitting linear factors X,Y..."; verbosity=verbosity, prefix=print_prefix)
        mf_fit_adapt_lr!(model; capacity=capacity, update_X=true, update_Y=true,
                                lr=lr, min_lr=0.05,
                                max_epochs=max_epochs, 
                                verbosity=verbosity, print_prefix=n_prefix,
                                history=history,
                                kwargs...)
    end

    if whiten
        v_println("Whitening X."; verbosity=verbosity, prefix=print_prefix)
        whiten!(model)
    end

    if svd_rotate
        v_println("Rotating factors via SVD."; verbosity=verbosity, prefix=print_prefix)
        rotate_by_svd!(model)
    end

    # Unfreeze all the layers
    unfreeze_layer!(model.matfac.col_transform,1:4) 
end


function basic_fit_reg_weight_eb!(model::PathMatFacModel; 
                                  capacity=Int(10e8), lr=1.0, max_epochs=1000, 
                                  verbosity=1, print_prefix="", history=nothing, kwargs...) 

    K = size(model.matfac.X,1)
    n_pref = string(print_prefix, "    ")
    freeze_reg!(model.matfac.col_transform_reg, 1:4)

    orig_X_reg = model.matfac.X_reg
    orig_Y_reg = model.matfac.Y_reg

    #model.matfac.X_reg = L2Regularizer(K, Float32(1/K))
    model.matfac.X_reg = X -> Float32(0.0) 
    model.matfac.Y_reg = construct_minimal_regularizer(model)

    # Fit the model without regularization (except the Bernoulli factors).
    # Whiten the embedding.
    v_println("Pre-fitting model with minimal regularization..."; prefix=print_prefix,
                                                                 verbosity=verbosity)
    fit_batch = isa(model.matfac.col_transform.layers[2], BatchScale)
    basic_fit!(model; fit_mu=true, fit_logsigma=true,
                      reweight_losses=true,
                      fit_batch=fit_batch, 
                      init_factors=true,
                      #fit_factors=true,
                      #vd_rotate=true,
                      whiten=true,
                      verbosity=verbosity, print_prefix=n_pref, 
                      capacity=capacity,
                      lr=lr, max_epochs=max_epochs,
                      history=history, kwargs...) 
     
    # Restore the regularizers; reweight the regularizers.
    v_println("Setting regularizer weights via Empirical Bayes..."; prefix=print_prefix, 
                                                                    verbosity=verbosity)
    unfreeze_reg!(model.matfac.col_transform_reg, 1:4)
    model.matfac.X_reg = orig_X_reg
    model.matfac.Y_reg = orig_Y_reg
    reweight_eb!(model.matfac.col_transform_reg, model.matfac.col_transform)
    reweight_eb!(model.matfac.X_reg, model.matfac.X)
    reweight_eb!(model.matfac.Y_reg, model.matfac.Y)
    history!(history; name="reweight_eb")

    # Re-fit the model with regularized factors
    v_println("Refitting with full regularization..."; prefix=print_prefix, 
                                                       verbosity=verbosity)
    basic_fit!(model; reweight_losses=true,
                      fit_factors=true, 
                      verbosity=verbosity, print_prefix=n_pref,
                      history=history,
                      capacity=capacity,
                      lr=lr, max_epochs=max_epochs, kwargs...) 

end


function fit_non_ard!(model::PathMatFacModel; fit_reg_weight="EB",
                                              lambda_max=nothing, 
                                              n_lambda=8,
                                              lambda_min_frac=1e-3,
                                              kwargs...)

    if fit_reg_weight=="EB"
        basic_fit_reg_weight_eb!(model; kwargs...)
    else
        fit_batch = isa(model.matfac.col_transform.layers[2], BatchScale)
        basic_fit!(model; fit_mu=true, fit_logsigma=true, reweight_losses=true,
                          fit_batch=fit_batch, 
                          #init_factors=true, 
                          fit_factors=true, 
                          #whiten=true, 
                          kwargs...)
    end
end


############################################
# Fit models with ARD regularization on Y

function fit_ard!(model::PathMatFacModel; max_epochs=1000, capacity=10^8,
                                          verbosity=1, print_prefix="",
                                          history=nothing,
                                          lr=1.0, lr_regress=1.0, lr_theta=1.0,
                                          kwargs...)

    n_pref = string(print_prefix, "    ")

    # First, we fit the model without regularization
    orig_X_reg = model.matfac.X_reg
    orig_ard = model.matfac.Y_reg

    K = size(model.matfac.Y, 1)
    v_println("Pre-fitting with minimal regularization..."; verbosity=verbosity,
                                                           prefix=print_prefix)
    #model.matfac.X_reg = L2Regularizer(K, Float32(1/K)) 
    model.matfac.X_reg = X -> Float32(0.0) 
    model.matfac.Y_reg = construct_minimal_regularizer(model)
 
    fit_batch = isa(model.matfac.col_transform.layers[2], BatchScale)
    basic_fit!(model; fit_batch=fit_batch,
                      fit_mu=true,
                      fit_logsigma=true,
                      init_factors=true,
                      reweight_losses=true,
                      svd_rotate=true,
                      whiten=true,
                      lr_regress=lr_regress, lr_theta=lr_theta,
                      verbosity=verbosity,
                      print_prefix=n_pref,
                      max_epochs=max_epochs,
                      capacity=capacity,
                      history=history,
                      kwargs...) 
    
    # Next, we put the ARD prior back in place and
    # continue fitting the model.
    model.matfac.X_reg = orig_X_reg
    reweight_eb!(model.matfac.X_reg, model.matfac.X)
    model.matfac.Y_reg = orig_ard
    reweight_eb!(model.matfac.Y_reg, model.matfac.Y)
    v_println("Reweighting column losses..."; verbosity=verbosity,
                                            prefix=print_prefix)
    reweight_col_losses!(model; capacity=capacity)
    v_println("Adjusting with ARD on Y..."; verbosity=verbosity,
                                            prefix=print_prefix)
    mf_fit_adapt_lr!(model; capacity=capacity, update_X=true, update_Y=true,
                            lr=lr, min_lr=0.01,
                            max_epochs=max_epochs,
                            verbosity=verbosity,
                            print_prefix=n_pref,
                            history=history,
                            kwargs...)

end



###################################################
# Fit models with geneset ARD regularization on Y
function fit_feature_set_ard!(model::PathMatFacModel; lr=1.0, 
                                                      capacity=10^8,
                                                      max_epochs=1000,
                                                      fsard_max_iter=10,
                                                      fsard_max_A_iter=1000,
                                                      fsard_n_lambda=20,
                                                      fsard_lambda_atol=1e-3,
                                                      fsard_frac_atol=1e-2,
                                                      fsard_A_prior_frac=0.5,
                                                      fsard_term_iter=5,
                                                      fsard_term_rtol=1e-5,
                                                      verbosity=1, print_prefix="",
                                                      history=nothing,
                                                      kwargs...)

    n_pref = string(print_prefix, "    ")

    # First, fit the model under a "vanilla" ARD regularizer.
    orig_reg = model.matfac.Y_reg
    model.matfac.Y_reg = ARDRegularizer(model.feature_views)
    v_println("##### Pre-fitting with vanilla ARD... #####"; verbosity=verbosity,
                                                 prefix=print_prefix)
    fit_ard!(model; max_epochs=max_epochs, capacity=capacity, lr=lr,
                    verbosity=verbosity, print_prefix=n_pref, history=history, kwargs...)

    # Next, put the FeatureSetARDReg back in place and
    # continue fitting the model.
    model.matfac.Y_reg = orig_reg
    model.matfac.Y_reg.A_opt.lambda .= set_lambda_max(model.matfac.Y_reg, 
                                                      model.matfac.Y)

    # Set alpha
    update_alpha!(model.matfac.Y_reg, model.matfac.Y)

    # For now we assess "convergence" via change in X
    X_old = deepcopy(model.matfac.X)

    for iter=1:fsard_max_iter

        # Fit A
        v_println("##### Featureset ARD outer iteration (", iter, ") #####"; verbosity=verbosity,
                                                                             prefix=print_prefix)
        update_A!(model.matfac.Y_reg, model.matfac.Y; target_frac=fsard_A_prior_frac,
                                                      max_epochs=fsard_max_A_iter, term_iter=50,
                                                      bin_search_max_iter=fsard_n_lambda,
                                                      bin_search_frac_atol=fsard_frac_atol,
                                                      bin_search_lambda_atol=fsard_lambda_atol,
                                                      print_prefix=n_pref,
                                                      print_iter=100,
                                                      verbosity=verbosity,
                                                      history=history) 

        # Re-fit the factors X, Y
        keep_history = (history != nothing)
        mf_fit_adapt_lr!(model; capacity=capacity, update_X=true, update_Y=true,
                                lr=lr, min_lr=0.01,
                                max_epochs=max_epochs,
                                verbosity=verbosity,
                                print_prefix=n_pref,
                                history=history)
            
        # Compute the relative change in X
        X_old .-= model.matfac.X
        X_old .= X_old.*X_old
        X_diff = sum(X_old)/sum(model.matfac.X.*model.matfac.X)

        v_println("##### (ΔX)^2/(X)^2 = ", X_diff, " #####"; verbosity=verbosity,
                                                         prefix=print_prefix)

        if X_diff < fsard_term_rtol
            v_println("##### (ΔX)^2/(X)^2 < ", fsard_term_rtol," #####"; verbosity=verbosity,
                                                                         prefix=print_prefix)
            v_println("##### Terminating."; verbosity=verbosity,
                                            prefix=print_prefix)
            break 
        end
        
        X_old .= model.matfac.X

        if iter == fsard_max_iter 
            v_println("##### Reached max iteration (", fsard_max_iter,"); #####"; verbosity=verbosity,
                                                                    prefix=print_prefix)
            v_println("##### Terminating."; verbosity=verbosity,
                                            prefix=print_prefix)
        end
    end

end


###############################################################
# MASTER FUNCTION
###############################################################
""" 
    fit!(model::PathMatFacModel; capacity=Int(10e8),
                                 verbosity=1, 
                                 lr=1.0,
                                 max_epochs=1000,
                                 fit_reg_weight="EB",
                                 lambda_max=1.0, 
                                 validation_frac=0.2,
                                 fsard_max_iter=10,
                                 fsard_max_A_iter=1000,
                                 fsard_n_lambda=20,
                                 fsard_lambda_atol=1e-3,
                                 fsard_frac_atol=1e-2,
                                 fsard_A_prior_frac=0.5,
                                 fsard_term_iter=5,
                                 fsard_term_rtol=1e-5,
                                 verbosity=1, print_prefix="",
                                 keep_history=false,
                                 kwargs...)
    
    Fit `model.matfac` on `model.data`. Keyword arguments control
    the fit procedure. By default, select regularizer weight via
    Empirical Bayes.
    
"""
function fit!(model::PathMatFacModel; lr=1.0,
                                      fit_reg_weight="EB",
                                      n_lambda=8,
                                      lambda_max=nothing,
                                      lambda_min_frac=1e-3, 
                                      keep_history=false,
                                      fit_joint=false,
                                      fsard_max_iter=10,
                                      fsard_max_A_iter=1000,
                                      fsard_n_lambda=20,
                                      fsard_lambda_atol=1e-3,
                                      fsard_frac_atol=1e-2,
                                      fsard_A_prior_frac=0.5,
                                      fsard_term_iter=5,
                                      fsard_term_rtol=1e-5,
                                      rel_tol=1e-5,
                                      abs_tol=1e-5,
                                      verbosity=1,
                                      print_prefix="",
                                      kwargs...)
  
    hist=nothing
    if keep_history
        hist = MutableLinkedList()
        global FIT_START_TIME = time()
        history!(hist; name="start")
    else
        global FIT_START_TIME = time()
    end   
 
    if isa(model.matfac.Y_reg, ARDRegularizer)
        fit_ard!(model; history=hist, verbosity=verbosity,
                                      print_prefix=print_prefix,
                                      rel_tol=rel_tol,
                                      abs_tol=abs_tol,
                                      kwargs...)
    elseif isa(model.matfac.Y_reg, FeatureSetARDReg)
        fit_feature_set_ard!(model; lr=lr,
                                    rel_tol=rel_tol,
                                    abs_tol=abs_tol,
                                    history=hist, 
                                    fsard_max_iter=fsard_max_iter,
                                    fsard_max_A_iter=fsard_max_A_iter,
                                    fsard_n_lambda=fsard_n_lambda,
                                    fsard_lambda_atol=fsard_lambda_atol,
                                    fsard_frac_atol=fsard_frac_atol,
                                    fsard_A_prior_frac=fsard_A_prior_frac,
                                    fsard_term_iter=fsard_term_iter,
                                    fsard_term_rtol=fsard_term_rtol,
                                    verbosity=verbosity,
                                    print_prefix=print_prefix,
                                    kwargs...)
    else
        fit_non_ard!(model; history=hist, 
                            rel_tol=rel_tol,
                            abs_tol=abs_tol,
                            fit_reg_weight=fit_reg_weight, 
                            lambda_max=lambda_max, 
                            n_lambda=n_lambda, 
                            lambda_min_frac=lambda_min_frac,
                            verbosity=verbosity,
                            print_prefix=print_prefix, 
                            kwargs...)
    end

    # Finally: allow the fitted parameters to share information
    if fit_joint
        freeze_layer!(model.matfac.col_transform, 1:3) # batch and column scale 
        unfreeze_layer!(model.matfac.col_transform, 4) # batch and column shift 
        v_println("Jointly adjusting parameters..."; verbosity=verbosity, prefix=print_prefix)
        n_prefix = string(print_prefix, "    ") 
        mf_fit_adapt_lr!(model; capacity=capacity, update_X=true, update_Y=true,
                                lr=lr,
                                update_col_layers=true,
                                max_epochs=max_epochs,
                                verbosity=verbosity,
                                print_prefix=n_prefix,
                                history=hist)
        unfreeze_layer!(model.matfac.col_transform, 1:3)
    end

    whiten!(model)
    reorder_by_importance!(model)
    history!(hist; name="reorder_factors")

    history!(hist; name="finish")
    hist = finalize_history(hist)

    return hist
end


