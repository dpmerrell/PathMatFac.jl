
export impute

function impute(model::MultiomicModel)

    Z = GPUMatFac.impute_values(model.matfac)

    return Z
end


