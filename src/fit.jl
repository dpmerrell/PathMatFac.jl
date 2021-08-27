
export fit!

function fit!(model::MultiomicModel; method="nesterov", kwargs...)

    GPUMatFac.fit!(model.matfac, model.omic_matrix; method=method, kwargs...)

end
