
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


function fit!(model::MultiomicModel, D::AbstractMatrix; kwargs...)

    # Permute the data columns to match the model's
    # internal ordering
    println("Rearranging data columns...")
    D = D[:,model.used_feature_idx]

    # First set some model parameters to the right ball-park
    initialize_params!(model, D)

    # Move model and data to GPU (if one exists); 
    # train the model; and then move back to CPU
    matfac_d = gpu(model.matfac)
    D = gpu(D)
    println("Fitting model to data...")
    fit!(matfac_d, D; kwargs...)
    D = nothing
    model.matfac = cpu(matfac_d)

    # Postprocess the model 
    postprocess!(model)

    return model
end


