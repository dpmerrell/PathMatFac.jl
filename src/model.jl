
export MultiomicModel

mutable struct MultiomicModel

    # Matrix factorization model
    matfac::MatFacModel

    # Information about data samples
    sample_ids::Vector{String}

    # Information about internal samples
    internal_sample_idx::Vector{Int}
    aug_sample_ids::Vector{String}

    # Information about data features
    feature_genes::Vector{String}
    feature_assays::Vector{String}

    # Information about internal features
    internal_feature_idx::Vector{Int}
    aug_feature_genes::Vector{String}
    aug_feature_assays::Vector{String}

end



