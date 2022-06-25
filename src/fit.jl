
import ScikitLearnBase: fit!


function sparse_diag(M::SparseMatrixCSC)
    return diag(M)
end

function sparse_diag(M::CuSparseMatrixCSC)
    return gpu(Vector(diag(cpu(M))))
end


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
    println(mean_vec)
    println(var_vec)

    # Initialize values of mu, logsigma
    model.matfac.col_transform.cshift.mu .= mean_vec
    logsigma = log.(sqrt.(var_vec))
    model.matfac.col_transform.cscale.logsigma .= logsigma 

    # Initialize "virtual" values in the mu/logsigma regularizers
    cshift_reg = model.matfac.col_transform_reg.cshift_reg
    model.matfac.col_transform_reg.cshift_reg.B_matrix .= -transpose( vec(transpose(mean_vec) * cshift_reg.AB[1]) ./ sparse_diag(cshift_reg.BB[1]) )
    
    cscale_reg = model.matfac.col_transform_reg.cscale_reg
    model.matfac.col_transform_reg.cscale_reg.B_matrix .= -transpose( vec(transpose(logsigma) * cscale_reg.AB[1]) ./ sparse_diag(cscale_reg.BB[1]) )
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

    # Permute the data columns to match the model's
    # internal ordering
    println("Rearranging data columns...")
    D = view(D, :, model.used_feature_idx)
    
    # Set some model parameters to the right ball-park
    initialize_params!(model, D; capacity=capacity)

    # train the model; and then move back to CPU
    println("Fitting model to data...")
    fit!(model.matfac, D; capacity=capacity, kwargs...)
    D = nothing

    # Postprocess the model 
    postprocess!(model)

    return model
end


