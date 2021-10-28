
export fit!

function fit!(model::MultiomicModel; method="nesterov", kwargs...)

    GPUMatFac.fit!(model.matfac, model.omic_matrix; method=method, kwargs...)

    result = Dict()

    result["R2"] = score_factors_R2(model.matfac, model.omic_matrix; 
                                    covariates=model.sample_covariates)
    
    return result
end
