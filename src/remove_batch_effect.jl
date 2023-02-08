

function remove_batch_effect(model::PathMatFacModel, Z::AbstractMatrix)

    # Apply the link functions
    tr_Z = MF.link(model.matfac.noise_model, Z)

    # Transform the data
    tr_Z = (tr_Z - model.matfac.col_transform.bshift.theta)
    tr_Z = tr_Z / exp(model.matfac.col_transform.bscale.logdelta)

    # Re-impute the transformed data 
    tr_Z = impute(model.matfac.noise_model, tr_Z)

    return tr_Z
end


