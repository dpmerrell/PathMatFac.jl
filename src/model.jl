
export MultiomicModel

mutable struct MultiomicModel

    # Matrix factorization model
    matfac::MatFacModel

    # Omic feature/pathway stuff
    feature_names::Vector      # vector of feature names
    pwy_graphs::Vector         # A vector of pathway graphs (weighted edge list representation)
    aug_feature_to_idx::Dict   # Maps from feature names to column indices in the MatFacModel

    # Sample stuff
    sample_ids::Vector         # vector of sample IDs
    sample_graph::Vector       # graph of relationships between samples (weighted edge list representation)
    aug_sample_to_idx::Dict    # Maps from sample IDs to row indices in the MatFacModel

    # Omic dataset
    omic_matrix::Union{Nothing,AbstractMatrix}

end



