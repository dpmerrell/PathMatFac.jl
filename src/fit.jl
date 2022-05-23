
import ScikitLearnBase: fit!


function initialize_params!(model::MultiomicModel, D::AbstractMatrix; 
                            loss_map=DEFAULT_ASSAY_LOSSES, verbose=true)

    if verbose
        println("Setting initial values for mu and sigma...")
    end

    model_losses = map(x->loss_map[x], model.data_assays[model.used_feature_idx]) 
    normal_columns = model_losses .== "normal"

    model.matfac.col_transform.cshift.mu[normal_columns] .= vec(mapslices(nanmean, 
                                                              D[:,normal_columns]; 
                                                              dims=1)
                                                   )

    model.matfac.col_transform.cscale.logsigma[normal_columns] .= log.(sqrt.(vec(mapslices(nanvar,
                                                                          D[:,normal_columns];
                                                                          dims=1)
                                                                     )
                                                                 )
                                                          )

end


function fit!(model::MultiomicModel, D::AbstractMatrix; kwargs...)

    # Permute the data columns to match the model's
    # internal ordering
    println("Rearranging data columns...")
    D = D[:,model.used_feature_idx]

    # First set some model parameters to the right ball-park
    initialize_params!(model, D)

    println("Fitting model to data...")
    fit!(model.matfac, D; kwargs...)

    return model
end


