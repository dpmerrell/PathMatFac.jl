

function impute(nm::MF.NormalNoise, Z::AbstractMatrix)
    return Z
end

function impute(nm::MF.BernoulliNoise, Z::AbstractMatrix)
    return MF.invlink(nm, Z) 
end

function impute(nm::MF.PoissonNoise, Z::AbstractMatrix)
    return MF.invlink(nm, Z)
end

function impute(nm::MF.OrdinalNoise, Z::AbstractMatrix)
    
    N_bins = length(nm.ext_thresholds)-1
    result = similar(Z)
    for i=1:N_bins
        relevant_idx = ((Z .> nm.ext_thresholds[i]) .& (Z .<= nm.ext_thresholds[i+1]))
        result[relevant_idx] .= i
    end

    return result
end

function impute(nm::MF.CompositeNoise, Z::AbstractMatrix, assays)
    X = similar(Z)

    for (idx, nm) in zip(nm.col_ranges, nm.noises)
        idx_assays = assays[idx]
        X[:,idx] .= impute(nm, Z[:,idx])
    end

    return X
end

function impute(model::MultiomicModel)
    Z = Base.invokelatest(model.matfac)
    used_assays = model.data_assays[model.used_feature_idx]
    Z = impute(model.matfac.noise_model, Z, used_assays)
    return Z
end



