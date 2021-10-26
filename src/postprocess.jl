

function compute_R2(y_pred, y_true)
    mu_y = nanmean(y_true)
    return 1.0 - nansum((y_pred .- y_true).^2.0) / nansum((y_true .- mu_y).^2.0)
end

function score_factors_R2(matfac_model, omic_data; covariates=nothing)

    K = size(matfac_model.X,1)
    scores = zeros(K)
    
    for k=1:K
        Z = GPUMatFac.impute_values(matfac_model; 
                                    covariates=covariates,
                                    factor_set=[k])

        scores[k] = compute_R2(Z, omic_data)
    end

    return scores 
end



