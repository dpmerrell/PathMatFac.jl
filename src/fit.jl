
import ScikitLearnBase: fit!


function normalize_factors!(model::MultiomicModel)

    Y = model.matfac.mp.Y

    Y_norms = sqrt.(sum(Y .* Y; dims=2))
    N = size(Y,2)
    target_Y_norm = N/100
    model.matfac.mp.Y .*= (target_Y_norm ./ Y_norms)

    model.pathway_weights .= vec(Matrix(Y_norms)) ./ (target_Y_norm) 

end


function fit!(model::MultiomicModel, D::AbstractMatrix; kwargs...)

    # Permute the data columns to match the model's
    # internal ordering
    internal_D = D[:,model.feature_idx]

    fit!(model.matfac, internal_D; kwargs...)

    normalize_factors!(model)

end


