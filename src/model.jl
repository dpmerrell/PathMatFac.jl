
export MultiomicModel

mutable struct MultiomicModel

    # Matrix factorization model
    matfac::MatFacModel

    ## Omic feature/pathway stuff
    #feature_names::Vector      # vector of feature names
    #aug_feature_to_idx::Dict   # Maps from feature names to column indices in the MatFacModel
    feature_genes::Vector
    feature_assays::Vector

    ## Sample stuff
    #sample_ids::Vector         # vector of sample IDs
    #aug_sample_to_idx::Dict    # Maps from sample IDs to row indices in the MatFacModel
    sample_ids::Vector
    sample_groups::Vector

    # Omic dataset
    omic_matrix::Union{Nothing,AbstractMatrix}

end



