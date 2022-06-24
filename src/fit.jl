
import ScikitLearnBase: fit!


function initialize_params!(model::MultiomicModel, D::AbstractMatrix;
                            capacity=Int(25e6), verbose=true,
                            loss_map=DEFAULT_ASSAY_LOSSES)

    if verbose
        println("Setting initial values for mu and sigma...")
    end
    
    M, N = size(D)
    batch_size = Int(round(capacity / N))
    model_losses = map(x->loss_map[x], model.data_assays[model.used_feature_idx])
    mean_vec, var_vec = MF.column_meanvar(D, batch_size)

    # Initialize values of mu, logsigma
    normal_columns = model_losses .== "normal"
    model.matfac.col_transform.cshift.mu[normal_columns] .= mean_vec[normal_columns]
    model.matfac.col_transform.cscale.logsigma[normal_columns] .= log.(sqrt.(var_vec[normal_columns]))

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


function fit!(model::MultiomicModel, D::AbstractMatrix; capacity=Int(25e6), kwargs...)

    # Move model and data to GPU (if one exists); 
    D = gpu(D)
    matfac_d = gpu(model.matfac)

    # Permute the data columns to match the model's
    # internal ordering
    println("Rearranging data columns...")
    D .= D[:,model.used_feature_idx]
    
    # Set some model parameters to the right ball-park
    initialize_params!(model, D; capacity=capacity)

    # train the model; and then move back to CPU
    println("Fitting model to data...")
    fit!(matfac_d, D; capacity=capacity, kwargs...)
    D = nothing
    model.matfac = cpu(matfac_d)

    # Postprocess the model 
    postprocess!(model)

    return model
end


