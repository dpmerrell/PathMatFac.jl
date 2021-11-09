
export fit!

function fit!(model::MultiomicModel; method="adagrad", kwargs...)

    GPUMatFac.fit!(model.matfac, model.omic_matrix; method=method, kwargs...)

end
