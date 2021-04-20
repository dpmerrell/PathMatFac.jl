
using LinearAlgebra, SparseArrays

export MatFacModel

mutable struct MatFacModel

    X::AbstractMatrix      # KxM "instance factor" matrix
    Y::AbstractMatrix      # KxN "feature factor" matrix

    losses::AbstractVector         # N-dim vector of losses; one per feature
    
    instance_reg_mats::AbstractVector{AbstractMatrix}  # K x (M x M)
    feature_reg_mats::AbstractVector{AbstractMatrix}  # K x (N x N)
    
end


function MatFacModel(instance_reg_mats::AbstractVector, 
                     feature_reg_mats::AbstractVector,
                     losses::AbstractVector)

    M = size(instance_reg_mats[1],1)
    N = size(feature_reg_mats[1],1)
    K = length(instance_reg_mats)
    
    X = 0.001 .* randn(K, M) ./ sqrt(K) 
    Y = 0.001 .* randn(K, N)

    return MatFacModel(X, Y, losses, instance_reg_mats,
                                     feature_reg_mats
                      )
end


