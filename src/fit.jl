
import StatsBase: fit!


# Change some of the default kwarg values
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


function construct_optimizer(model, lr)
    return Flux.Optimise.AdaGrad(lr)
end


function fit!(model::PathMatFacModel; capacity::Int=10^8, verbosity=1, 
                                      lr=0.25, kwargs...)

    K,N = size(model.matfac.Y)

    # Compute the column M-estimates.
    println("Computing column M-estimates...") 
    M_estimates = MF.compute_M_estimates(model.matfac, model.data; 
                                         capacity=capacity, verbosity=verbosity-2, 
                                         lr=1.0, max_epochs=500, rel_tol=1e-9)

    # Choose column scales in a way that encourages
    # columns of Y to have similar magnitudes.
    println("Computing column scale parameters...")
    col_scales = MF.link_scale(model.matfac.noise_model, model.data; capacity=capacity) 
    model.matfac.col_transform.layers[1].logsigma .= log.(col_scales ./ sqrt(K))
 
    # Freeze the column scales.
    freeze_layer!(model.matfac.col_transform, 1) # Col scales

    # Reweight the column losses.
    println("Reweighting column losses...")
    ssq_grads = vec(MF.batched_column_ssq_grads(model.matfac, M_estimates, model.data; 
                                                capacity=capacity)
                   )
    M = MF.column_nonnan(model.data)
    weights = sqrt.(M ./ ssq_grads)
    weights[ (!isfinite).(weights) ] .= 1
    weights = map(x -> max(x, 1e-2), weights)
    weights = map(x -> min(x, 10.0), weights)
    MF.set_weight!(model.matfac.noise_model, vec(weights))

    # Freeze the column shifts
    model.matfac.col_transform.layers[2].mu .= vec(M_estimates)
    freeze_layer!(model.matfac.col_transform, 2)

    # Construct an optimizer. Its state will
    # persist between calls to `mf_fit!`
    opt = construct_optimizer(model, lr) 

    # If the model has batch parameters:
    if length(model.matfac.col_transform.layers) > 2
        # Freeze the batch scale parameters... 
        # not sure if they're even necessary
        freeze_layer!(model.matfac.col_transform, 3)

        # Fit the batch shift parameters.
        println("Fitting batch parameters...")
        mf_fit!(model; capacity=capacity, update_col_layers=true, 
                       lr=0.1, max_epochs=500, verbosity=verbosity-1,
                       opt=opt)
    end

    # Fit the factors (X,Y), and their regularizers.
    println("Fitting linear factors.")
    mf_fit!(model; capacity=capacity, update_X=true, update_Y=true,
                                      update_X_reg=true, update_Y_reg=true,
                                      opt=opt,
                                      verbosity=verbosity,
                                      kwargs...)

    # Finally: jointly fit X, Y, column shifts, and batch shifts
    unfreeze_layer!(model.matfac.col_transform, 2) # Col shifts
    if length(model.matfac.col_transform.layers) > 2
        unfreeze_layer!(model.matfac.col_transform, 4) # Batch shifts
    end
    println("Jointly adjusting parameters.")
    mf_fit!(model; capacity=capacity, update_X=true, update_Y=true,
                                      update_col_layers=true,
                                      opt=opt,
                                      verbosity=verbosity,
                                      kwargs...)

    unfreeze_layer!(model.matfac.col_transform,1) 
    if length(model.matfac.col_transform.layers) > 2
        unfreeze_layer!(model.matfac.col_transform,3) 
    end 
end



