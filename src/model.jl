
export MultiomicModel

mutable struct MultiomicModel

    # Matrix factorization model
    matfac::MatFacModel

    ## Omic feature/pathway stuff
    original_genes::Vector
    original_assays::Vector
    augmented_genes::Vector
    augmented_assays::Vector
    feature_to_idx::Dict   # Maps from features to column indices in the MatFacModel

    ## Sample stuff
    original_samples::Vector
    original_groups::Vector
    augmented_samples::Vector
    sample_to_idx::Dict    # Maps from sample IDs to row indices in the MatFacModel

    # Omic dataset
    omic_matrix::Union{Nothing,AbstractMatrix}
    sample_covariates::Union{Nothing,AbstractMatrix}
end



