
import StatsBase: fit!


######################################################################
# Wrapper for MatFac.fit!
######################################################################

function mf_fit!(model::PathMatFacModel; scale_column_losses=false,
                                         update_X=false,
                                         update_Y=false,
                                         update_row_layers=false,
                                         update_col_layers=false,
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
                                               verbosity=1, kwargs...)
    
    M_estimates = MF.compute_M_estimates(model.matfac, model.data;
                                         capacity=capacity, max_epochs=max_epochs,
                                         opt=opt, verbosity=verbosity) 
    model.matfac.col_transform.layers[2].mu .= vec(M_estimates)

end


function init_logsigma!(model::PathMatFacModel; capacity=Int(1e8))
   
    K = size(model.matfac.X, 1) 
    col_scales = MF.link_scale(model.matfac.noise_model, model.data; capacity=capacity) 
    model.matfac.col_transform.layers[1].logsigma .= log.(col_scales ./ sqrt(K))

end


function reweight_losses!(model::PathMatFacModel; capacity=Int(1e8))
    

    # Freeze the column scales.
    freeze_layer!(model.matfac.col_transform, 1) # Col scales

    # Set the noise model's weights to 1
    N = size(model.data, 2)
    one_vec = similar(model.data,N)
    one_vec .= 1
    MF.set_weight!(model.matfac.noise_model, one_vec)

    # Reweight the column losses.
    M_estimates = transpose(model.matfac.col_transform.layers[2].mu) 
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
    MF.set_weight!(model.matfac.noise_model, vec(weights))

end


#######################################################################
# FITTING PROCEDURES
#######################################################################

################################
# Procedures for simple models

function basic_fit!(model::PathMatFacModel; fit_mu=true, fit_logsigma=true,
                                            reweight_losses=true,
                                            fit_batch=true, fit_factors=true,
                                            fit_joint=true,
                                            verbosity=1, capacity=Int(1e8),
                                            opt=nothing, lr=0.25, max_epochs=1000, 
                                            kwargs...)

    K,N = size(model.matfac.Y)
    
    # If an optimizer doesn't exist, then construct one. 
    # Its state will persist between calls to `mf_fit!`
    if opt == nothing
        opt = construct_optimizer(model, lr)
    end

    if fit_mu
        # Initialize mu with the column M-estimates.
        println("Computing column M-estimates...")
        init_mu!(model, opt; capacity=capacity, 
                             verbosity=verbosity-1,
                             max_epochs=500)
    end

    if fit_logsigma
        # Choose column scales in a way that encourages
        # columns of Y to have similar magnitudes.
        println("Computing column scale parameters...")
        init_logsigma!(model; capacity=capacity)
    end

    if reweight_losses 
        # Reweight the column losses so that they exert 
        # similar gradients on X. This also freezes logsigma
        println("Reweighting column losses...")
        reweight_losses!(model; capacity=capacity)
    end

    # Freeze the column shifts
    freeze_layer!(model.matfac.col_transform, 2)

    # If the model has batch parameters:
    if (fit_batch & isa(model.matfac.col_transform.layers[4], BatchShift))
        # Freeze the batch scale parameters... 
        # not sure if they're even necessary
        freeze_layer!(model.matfac.col_transform, 3)

        # Fit the batch shift parameters.
        println("Fitting batch parameters...")
        mf_fit!(model; update_col_layers=true, capacity=capacity,
                       max_epochs=500, verbosity=verbosity-1,
                       opt=opt)
    end

    if fit_factors
        # Fit the factors X,Y
        println("Fitting linear factors.")
        mf_fit!(model; capacity=capacity, update_X=true, update_Y=true,
                       opt=opt, max_epochs=max_epochs, verbosity=verbosity)
    end

    if fit_joint
        # Finally: jointly fit X, Y, and batch shifts
        unfreeze_layer!(model.matfac.col_transform, 4) # batch effects 
        println("Jointly adjusting parameters.")
        mf_fit!(model; capacity=capacity, update_X=true, update_Y=true,
                                          update_col_layers=true,
                                          opt=opt,
                                          verbosity=verbosity,
                                          max_epochs=max_epochs)
    end

    # Unfreeze all the layers
    unfreeze_layer!(model.matfac.col_transform,1:4) 
end


function basic_fit_reg_weight!(model::PathMatFacModel; 
                               verbosity=1, capacity=Int(1e8),
                               validation_frac=0.2,
                               lr=0.25, opt=nothing, 
                               lambda_max=1.0, 
                               n_lambda=8,
                               lambda_min=1e-6,
                               kwargs...)
    if opt == nothing
        opt = construct_optimizer(model, lr)
    end

    # Hold out random validation set
    M, N = size(model.data)
    model.data = cpu(model.data)
    val_idx = sprand(Bool, M, N, validation_frac)
    true_validation = model.data[val_idx]
    model.data[val_idx] .= NaN
    model = gpu(model)

    # Loop over a set of weights (in decreasing order)
    lambda_vec = exp.(collect(range(log(lambda_max), 
                                    log(lambda_min); 
                                    length=n_lambda)))
    orig_X_reg = model.matfac.X_reg
    orig_Y_reg = model.matfac.Y_reg
    orig_layer_reg = model.matfac.col_transform_reg
    best_val_loss = Inf
    best_model = deepcopy(cpu(model))
    for (i,lambda) in enumerate(lambda_vec)

        println(string("Outer loop: ", i,"/",n_lambda,";\tλ=",lambda))
        # Set the regularizer weights for X, Y, and layers
        model.matfac.X_reg = (X -> lambda*orig_X_reg(X))
        model.matfac.Y_reg = (Y -> lambda*orig_Y_reg(Y))
        model.matfac.col_transform_reg = (layers -> lambda*orig_layer_reg(layers))

        # (re-)fit with the new weights
        basic_fit!(model; verbosity=verbosity, capacity=capacity, opt=opt)

        # Compute loss on the test data
        model_cpu = deepcopy(cpu(model))
        train_loss = MF.batched_data_loss(model_cpu.matfac, model_cpu.data; capacity=capacity)
        model_cpu.data[val_idx] .= true_validation
        train_val_loss = MF.batched_data_loss(model_cpu.matfac, model_cpu.data; capacity=capacity)
        val_loss = train_val_loss - train_loss
        println(("Validation loss: ", val_loss))
 
        # Check whether this is an improvement
        if val_loss < best_val_loss
            best_val_loss = val_loss
            best_model = deepcopy(model_cpu)
        end 
    end

    model.matfac = best_model.matfac
end


function fit_non_ard!(model::PathMatFacModel; fit_reg_weight=false,
                                              lambda_max=1.0, 
                                              n_lambda=8,
                                              lambda_min=1e-6,
                                              kwargs...)

    if fit_reg_weight
        basic_fit_reg_weight!(model; lambda_max=lambda_max,
                                     n_lambda=n_lambda,
                                     lambda_min=lambda_min,
                                     kwargs...)
    else
        basic_fit!(model; kwargs...)
    end
end


############################################
# Fit models with ARD regularization on Y

function fit_ard!(model::PathMatFacModel; lr=0.1, opt=nothing, 
                                          kwargs...)

    K, M = size(model.matfac.X)
    Kd2M = 0.5*K/M # This weight counteracts the multiplier
                   # introduced by MF.fit!

    if opt == nothing
        opt = construct_optimizer(model, lr)
    end

    # First, we fit the model using an L2 regularizer on Y
    orig_ard = model.matfac.Y_reg
    matfac.Y_reg = Y -> Kd2M*sum(Y.*Y)

    basic_fit!(model; opt=opt, kwargs...)
    
    # Next, we put the ARD prior back in place and
    # continue fitting the model.
    basic_fit!(model; opt=opt, 
                      fit_mu=false, fit_logsigma=false, fit_batch=false,
                      kwargs...)
    

end


function fit_ard_with_reg_weights!(model::PathMatFacModel)

end


###################################################
# Fit models with geneset ARD regularization on Y
function fit_geneset_ard!(model::PathMatFacModel)

end


###############################################################
# MASTER FUNCTION
###############################################################
""" 
    fit!(model::PathMatFacModel; capacity=Int(1e8),
                                 verbosity=1, 
                                 lr=0.25,
                                 max_epochs=1000,
                                 opt=nothing, 
                                 fit_reg_weight=false,
                                 lambda_max=1.0, 
                                 n_lambda=8,
                                 lambda_min=1e-6,
                                 validation_frac=0.2,
                                 kwargs...)
    
    Fit `model.matfac` on `model.data`. Keyword arguments control
    the fit procedure. By default, do *not* select regularizer weight.
    
"""
function fit!(model::PathMatFacModel; kwargs...)
    
    if isa(model.matfac.Y_reg, ARDRegularizer)
        fit_ard!(model; kwargs...)
    #TODO: ADD CONDITION FOR GENESET ARD
    else
        fit_non_ard!(model; kwargs...)
    end

end


