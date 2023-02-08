

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

function impute(nm::MF.CompositeNoise, Z::AbstractMatrix)
    X = similar(Z)

    for (idx, nm) in zip(nm.col_ranges, nm.noises)
        X[:,idx] .= impute(nm, Z[:,idx])
    end

    return X
end

function impute(model::PathMatFacModel; include_batch_effects=false)

    matfac = model.matfac

    # Apply column scales and shifts
    Z = transpose(matfac.X) * matfac.Y
    Z = Base.invokelatest(matfac.col_transform.cscale, Z)
    Z = Base.invokelatest(matfac.col_transform.cshift, Z)

    # Apply the batch effects if specified
    if include_batch_effects
        Z = Base.invokelatest(matfac.row_transform.bscale, Z)
        Z = Base.invokelatest(matfac.row_transform.bshift, Z)
    end

    # Finally, impute the data 
    Z = impute(model.matfac.noise_model, Z)

    return Z
end



