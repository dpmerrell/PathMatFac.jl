
using LinearAlgebra, SparseArrays


mutable struct MatFacModel

    X::AbstractMatrix{Number}      # KxM "instance factor" matrix
    Y::AbstractMatrix{Number}      # KxN "feature factor" matrix

    losses::AbstractVector         # N-dim vector of losses; one per feature
    
    instance_reg_mats::AbstractVector{AbstractMatrix}  # K x (M x M)
    feature_reg_mats::AbstractVector{AbstractMatrix}  # K x (N x N)
    
end


function MatFacModel(instance_reg_mats::AbstractVector{AbstractMatrix}, 
                     feature_reg_mats::AbstractVector{AbstractMatrix})

    M = size(instance_reg_mats[1],1)
    N = size(feature_reg_mats[1],1)
    K = length(instance_reg_mats)
    
    X = 0.01 .* randn(K, M) ./ sqrt(K) 
    Y = 0.001 .* randn(K, N)

    return MatFacModel(X, Y, losses, instance_reg_mats,
                                     feature_reg_mats,
                                     obs_matrix
                      )
end


