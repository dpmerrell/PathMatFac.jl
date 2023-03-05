
import StatsBase: fit!


######################################################################
# Wrapper for MatFac.fit!
######################################################################

function mf_fit!(model::PathMatFacModel; scale_column_losses=false,
                                         update_X=false,
                                         update_Y=false,
                                         update_row_layers=false,
                                         update_col_layers=false,
                                         reg_relative_weighting=false,
                                         update_X_reg=false,
                                         update_Y_reg=false,
                                         update_row_layers_reg=false,
                                         update_col_layers_reg=false,
                                         kwargs...)

    # Change some of the default kwarg values

    MF.fit!(model.matfac, model.data; scale_column_losses=scale_column_losses, 
                                      update_X=update_X,
                                      update_Y=update_Y,
                                      update_row_layers=update_row_layers,
                                      update_col_layers=update_col_layers,
                                      update_X_reg=update_X_reg,
                                      update_Y_reg=update_Y_reg,
                                      update_row_layers_reg=update_row_layers_reg,
                                      update_col_layers_reg=update_col_layers_reg,
                                      kwargs...)
end



########################################################################
# PARAMETER INITIALIZATION
########################################################################

function construct_optimizer(model, lr)
    return Flux.Optimise.AdaGrad(lr)
end

function init_mu!(model::PathMatFacModel, opt; capacity=Int(1e8), max_epochs=500,
                                               verbosity=1, print_prefix="", kwargs...)
    
    M_estimates = MF.compute_M_estimates(model.matfac, model.data;
                                         capacity=capacity, max_epochs=max_epochs,
                                         opt=opt, verbosity=verbosity, print_prefix=print_prefix) 
    model.matfac.col_transform.layers[3].mu .= vec(M_estimates)

end


function init_theta!(model::PathMatFacModel, opt; capacity=Int(1e8), max_epochs=500,
                                                  verbosity=1, print_prefix="", kwargs...)
    # Freeze everything except batch shift... 
    freeze_layer!(model.matfac.col_transform, 1:3)

    mf_fit!(model; update_col_layers=true, capacity=capacity,
                   max_epochs=500, opt=opt,
                   verbosity=verbosity-1,
                   print_prefix=print_prefix)
    
    unfreeze_layer!(model.matfac.col_transform, 1:3)

end

function init_logsigma!(model::PathMatFacModel; capacity=Int(1e8))
  
    # If applicable, account for batch shift when we compute these
    # column scales 
    latent_map_fn = (m,D) -> D
    if isa(model.matfac.col_transform.layers[4], BatchShift)
        latent_map_fn=(m,d)-> d - m.col_transform.layers[4].theta
    end
 
    col_scales = MF.batched_link_scale(model.matfac, model.data; 
                                       capacity=capacity,
                                       latent_map_fn=latent_map_fn) 
    
    K = size(model.matfac.X, 1) 
    model.matfac.col_transform.layers[1].logsigma .= log.(col_scales ./ sqrt(K))

end


function reweight_col_losses!(model::PathMatFacModel; capacity=Int(1e8))
    

    # Freeze the model layers.
    freeze_layer!(model.matfac.col_transform, [1,2,4])
    unfreeze_layer!(model.matfac.col_transform, 3) 

    # Set the noise model's weights to 1
    N = size(model.data, 2)
    one_vec = similar(model.data,N)
    one_vec .= 1
    MF.set_weight!(model.matfac.noise_model, one_vec)

    # Compute the RMS norm of the loss gradient w.r.t. X, for each column
    M_estimates = transpose(model.matfac.col_transform.layers[3].mu) 
    ssq_grads = vec(MF.batched_column_ssq_grads(model.matfac,
                                                M_estimates,
                                                model.data; 
                                                capacity=capacity)
                   )
    M_vec = MF.column_nonnan(model.data)
    weights = sqrt.(M_vec ./ ssq_grads)
    weights[ (!isfinite).(weights) ] .= 1
    weights = map(x -> max(x, 1e-2), weights)
    weights = map(x -> min(x, 10.0), weights)
    
    # Set the weights
    MF.set_weight!(model.matfac.noise_model, vec(weights))

    # Unfreeze the model layers
    unfreeze_layer!(model.matfac.col_transform, [1,2,4])
end

#######################################################################
# POSTPROCESSING FUNCTIONS
#######################################################################

# "Whiten" the embedding X, such that std(X_k) = 1 for each k;
# reallocating its variance to Y.
function whiten!(model::PathMatFacModel)
    X_std = std(model.matfac.X, dims=2)
    model.matfac.X ./= X_std
    model.matfac.Y .*= X_std
end


#######################################################################
# FITTING PROCEDURES
#######################################################################

################################
# Procedures for simple models

function basic_fit!(model::PathMatFacModel; fit_mu=false, fit_logsigma=false,
                                            reweight_losses=false,
                                            fit_batch=false, fit_factors=false,
                                            fit_joint=false,
                                            whiten=false,
                                            capacity=Int(1e8),
                                            opt=nothing, lr=0.05, max_epochs=1000, 
                                            verbosity=1, print_prefix="",
                                            kwargs...)

    n_prefix = string(print_prefix, "    ")
    K,N = size(model.matfac.Y)
    
    # If an optimizer doesn't exist, then construct one. 
    # Its state will persist between calls to `mf_fit!`
    if opt == nothing
        opt = construct_optimizer(model, lr)
    end

    if fit_mu
        # Initialize mu with the column M-estimates.
        v_println("Fitting column shifts..."; verbosity=verbosity,
                                              prefix=print_prefix)
        init_mu!(model, opt; capacity=capacity, 
                             max_epochs=500,
                             verbosity=verbosity-1,
                             print_prefix=n_prefix)
    end

    # If the model has batch parameters:
    if (fit_batch & isa(model.matfac.col_transform.layers[4], BatchShift))
        # Fit the batch shift parameters.
        v_println("Fitting batch parameters..."; verbosity=verbosity,
                                                 prefix=print_prefix)
        init_theta!(model, opt; capacity=capacity, 
                                max_epochs=500,
                                verbosity=verbosity-1,
                                print_prefix=n_prefix)
        
    end

    # Choose column scales in a way that encourages
    # columns of Y to have similar magnitudes.
    if fit_logsigma
        v_println("Fitting column scales..."; verbosity=verbosity,
                                              prefix=print_prefix)
        init_logsigma!(model; capacity=capacity)
    end

    # Reweight the column losses so that they exert 
    # similar gradients on X
    if reweight_losses 
        v_println("Reweighting column losses..."; verbosity=verbosity,
                                                  prefix=print_prefix)
        reweight_col_losses!(model; capacity=capacity)
    end

    # Fit the factors X,Y
    if fit_factors
        v_println("Fitting linear factors..."; verbosity=verbosity, prefix=print_prefix)
        mf_fit!(model; capacity=capacity, update_X=true, update_Y=true,
                       opt=opt, max_epochs=max_epochs, 
                       verbosity=verbosity, print_prefix=n_prefix,
                       kwargs...)
    end

    # Finally: jointly fit X, Y, and batch shifts
    if fit_joint
        freeze_layer!(model.matfac.col_transform, 1:2) # batch and column scale 
        unfreeze_layer!(model.matfac.col_transform, 3:4) # batch and column shift 
        v_println("Jointly adjusting parameters..."; verbosity=verbosity, prefix=print_prefix)
        mf_fit!(model; capacity=capacity, update_X=true, update_Y=true,
                                          update_col_layers=true,
                                          opt=opt,
                                          max_epochs=max_epochs,
                                          verbosity=verbosity,
                                          print_prefix=n_prefix,
                                          kwargs...)
    end

    if whiten
        v_println("Whitening X."; verbosity=verbosity, prefix=print_prefix)
        whiten!(model)
    end

    # Unfreeze all the layers
    unfreeze_layer!(model.matfac.col_transform,1:4) 
end


function basic_fit_reg_weight_eb!(model::PathMatFacModel; 
                                  capacity=Int(1e8), opt=nothing, lr=0.05, max_epochs=1000, 
                                  verbosity=1, print_prefix="", kwargs...) 

    n_pref = string(print_prefix, "    ")

    freeze_reg!(model.matfac.col_transform_reg, 1:4)

    orig_X_reg = model.matfac.X_reg
    model.matfac.X_reg = X->0.0

    orig_Y_reg = model.matfac.Y_reg
    model.matfac.Y_reg = Y->0.0

    # Fit the model without regularization. 
    # Whiten the embedding.
    v_println("Fitting unregularized model..."; prefix=print_prefix,
                                                verbosity=verbosity)
    basic_fit!(model; fit_mu=true, fit_logsigma=true,
                      reweight_losses=true,
                      fit_batch=true, fit_factors=true,
                      whiten=true,
                      verbosity=verbosity, print_prefix=n_pref, 
                      capacity=capacity,
                      opt=opt, lr=lr, max_epochs=max_epochs, kwargs...) 
    
    # Restore the regularizers; reweight the regularizers.
    v_println("Setting regularizer weights via Empirical Bayes..."; prefix=print_prefix, 
                                                                    verbosity=verbosity)
    unfreeze_reg!(model.matfac.col_transform_reg, 1:4)
    model.matfac.X_reg = orig_X_reg
    model.matfac.Y_reg = orig_Y_reg
    reweight_eb!(model.matfac.col_transform_reg, model.matfac.col_transform)
    reweight_eb!(model.matfac.X_reg, model.matfac.X)
    reweight_eb!(model.matfac.Y_reg, model.matfac.Y)

    # Re-fit the model with regularized factors
    v_println("Refitting with regularization..."; prefix=print_prefix, 
                                                  verbosity=verbosity)
    basic_fit!(model; fit_factors=true, fit_joint=true,
                      verbosity=verbosity, print_prefix=n_pref,
                      capacity=capacity,
                      opt=opt, lr=lr, max_epochs=max_epochs, kwargs...) 

end


function basic_fit_reg_weight_crossval!(model::PathMatFacModel; 
                                        capacity=Int(1e8),
                                        validation_frac=0.1,
                                        lr=0.05, opt=nothing, 
                                        use_gpu=true,
                                        lambda_max=1.0, 
                                        n_lambda=8,
                                        lambda_min_frac=1e-3,
                                        verbosity=1, print_prefix="", 
                                        kwargs...)
    n_pref = string(print_prefix, "    ")

    if opt == nothing
        opt = construct_optimizer(model, lr)
    end

    # Hold out random validation set
    M, N = size(model.data)
    model.data = cpu(model.data)
    val_idx = sprand(Bool, M, N, validation_frac)
    true_validation = model.data[val_idx]
    model.data[val_idx] .= NaN
    if use_gpu
        model = gpu(model)
    end

    v_println("Pre-fitting model layers..."; verbosity=verbosity,
                                             prefix=print_prefix)
    basic_fit!(model; capacity=capacity, opt=opt,
                      fit_mu=true, fit_logsigma=true,
                      reweight_losses=true,
                      fit_batch=true, 
                      print_prefix=n_pref, verbosity=verbosity)

    # Loop over a set of weights (in decreasing order)
    lambda_min = lambda_min_frac*lambda_max
    lambda_vec = exp.(collect(range(log(lambda_max), 
                                    log(lambda_min); 
                                    length=n_lambda)))
    orig_X_reg = model.matfac.X_reg
    orig_Y_reg = model.matfac.Y_reg
    orig_layer_reg = model.matfac.col_transform_reg
    best_val_loss = Inf
    best_model = deepcopy(cpu(model))
    for (i,lambda) in enumerate(lambda_vec)

        v_println("Outer loop: ", i,"/",n_lambda,";\tλ=",lambda; verbosity=verbosity,
                                                                 prefix=print_prefix)
        # Set the regularizer weights for X, Y, and layers
        model.matfac.X_reg = (X -> lambda*orig_X_reg(X))
        model.matfac.Y_reg = (Y -> lambda*orig_Y_reg(Y))
        model.matfac.col_transform_reg = (layers -> lambda*orig_layer_reg(layers))

        # (re-)fit with the new weights
        basic_fit!(model; capacity=capacity, opt=opt,
                          fit_factors=true,
                          print_prefix=n_pref, verbosity=verbosity)

        # Compute loss on the test data
        model_cpu = deepcopy(cpu(model))
        train_loss = MF.batched_data_loss(model_cpu.matfac, model_cpu.data; capacity=capacity)
        model_cpu.data[val_idx] .= true_validation
        train_val_loss = MF.batched_data_loss(model_cpu.matfac, model_cpu.data; capacity=capacity)
        val_loss = train_val_loss - train_loss
        v_println("Validation loss: ", val_loss; verbosity=verbosity,
                                                 prefix=n_pref)
 
        # Check whether this is an improvement
        if val_loss < best_val_loss
            best_val_loss = val_loss
            best_model = deepcopy(model_cpu)
        end 
    end

    model.matfac = best_model.matfac
end


function fit_non_ard!(model::PathMatFacModel; fit_reg_weight="eb",
                                              lambda_max=1.0, 
                                              n_lambda=8,
                                              lambda_min=1e-6,
                                              kwargs...)

    if fit_reg_weight=="EB"
        basic_fit_reg_weight_eb!(model; kwargs...)
    elseif fit_reg_weight=="crossval"
        basic_fit_reg_weight_crossval!(model; lambda_max=lambda_max,
                                              n_lambda=n_lambda,
                                              lambda_min=lambda_min,
                                              kwargs...)
    else
        basic_fit!(model; fit_mu=true, fit_logsigma=true, reweight_losses=true,
                          fit_batch=true, fit_factors=true, fit_joint=true,
                          whiten=true, 
                          kwargs...)
    end
end


############################################
# Fit models with ARD regularization on Y

function fit_ard!(model::PathMatFacModel; lr=0.05, opt=nothing, 
                                          verbosity=1, print_prefix="", 
                                          kwargs...)

    n_pref = string(print_prefix, "    ")

    if opt == nothing
        opt = construct_optimizer(model, lr)
    end

    # First, we fit the model in an Empirical Bayes fashion
    # with an L2 regularizer on Y
    orig_ard = model.matfac.Y_reg

    K = size(model.matfac.Y, 1)
    model.matfac.Y_reg = L2Regularizer(K, 1.0)
    v_println("Pre-training with Empirical Bayes L2 regularization..."; verbosity=verbosity,
                                                                        prefix=print_prefix)
    basic_fit_reg_weight_eb!(model; opt=opt, verbosity=verbosity,
                                             print_prefix=n_pref,
                                             kwargs...) 
    
    # Next, we put the ARD prior back in place and
    # continue fitting the model.
    v_println("Training with ARD on Y..."; verbosity=verbosity,
                                           prefix=print_prefix)
    model.matfac.Y_reg = orig_ard
    basic_fit!(model; opt=opt, fit_factors=true, 
                               verbosity=verbosity,
                               print_prefix=n_pref,
                               kwargs...)

end



###################################################
# Fit models with geneset ARD regularization on Y
function fit_feature_set_ard!(model::PathMatFacModel; lr=0.05, opt=nothing,
                                                      kwargs...)

    if opt == nothing
        opt = construct_optimizer(model, lr)
    end

    # First, fit the model under a "vanilla" ARD regularizer.
    orig_reg = model.matfac.Y_reg
    model.matfac.Y_reg = ARDRegularizer()
    fit_ard!(model; opt=opt, kwargs...)

    # Next, put the FeatureSetARDReg back in place and
    # continue fitting the model.
    model.matfac.Y_reg = orig_reg
    basic_fit!(model; opt=opt, fit_factors=true, fit_joint=true,
                               kwargs...)
end


###############################################################
# MASTER FUNCTION
###############################################################
""" 
    fit!(model::PathMatFacModel; capacity=Int(1e8),
                                 verbosity=1, 
                                 lr=0.05,
                                 max_epochs=1000,
                                 opt=nothing, 
                                 fit_reg_weight="EB",
                                 lambda_max=1.0, 
                                 n_lambda=8,
                                 lambda_min=1e-6,
                                 validation_frac=0.2,
                                 kwargs...)
    
    Fit `model.matfac` on `model.data`. Keyword arguments control
    the fit procedure. By default, select regularizer weight via
    Empirical Bayes.
    
"""
function fit!(model::PathMatFacModel; opt=nothing, lr=0.05, kwargs...)
  
    if opt == nothing
        opt = construct_optimizer(model, lr) 
    end 
    
    if isa(model.matfac.Y_reg, ARDRegularizer)
        fit_ard!(model; opt=opt, kwargs...)
    elseif isa(model.matfac.Y_reg, FeatureSetARDReg)
        fit_feature_set_ard!(model; opt=opt, kwargs...)
    else
        fit_non_ard!(model; opt=opt, kwargs...)
    end

end


