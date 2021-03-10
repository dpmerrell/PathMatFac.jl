

export RRGLRM, scale_regularizer, 
       Loss, QuadLoss, LogisticLoss,
       Regularizer, ObsArray,
       fixed_latent_features


import LowRankModels: AbstractGLRM, Loss, QuadLoss, LogisticLoss, 
                      Regularizer, ObsArray, embedding_dim, 
                      equilibrate_variance!, add_offset!,
                      sort_observations,
                      fixed_latent_features


mutable struct RRGLRM <: AbstractGLRM
    A                            # The data table
    losses::Array{Loss,1}        # array of loss functions
    rx::Array{Regularizer,1}     # Array of regularizers to be applied to each column of X
    ry::Array{Regularizer,1}     # Array of regularizers to be applied to each column of Y
    k::Int                       # Desired rank
    observed_features::ObsArray  # for each example, an array telling which features were observed
    observed_examples::ObsArray  # for each feature, an array telling in which examples the feature was observed
    X::AbstractArray{Float64,2}  # Representation of data in low-rank space. A ≈ X'Y
    Y::AbstractArray{Float64,2}  # Representation of features in low-rank space. A ≈ X'Y
    feature_ids::Vector
    feature_graphs #::Vector{ElUgraph}
    instance_ids::Vector
end


# usage notes:
# * providing argument `obs` overwrites arguments `observed_features` and `observed_examples`
# * offset and scale are *false* by default to avoid unexpected behavior
# * convenience methods for calling are defined in utilities/conveniencemethods.jl
function RRGLRM(A, losses::Array, rx::Array, ry::Array, k::Int;
# the following tighter definition fails when you form an array of a tighter subtype than the abstract type, eg Array{QuadLoss,1}
# function GLRM(A::AbstractArray, losses::Array{Loss,1}, rx::Array{Regularizer,1}, ry::Array{Regularizer,1}, k::Int;
              X = randn(k,size(A,1)), Y = randn(k,embedding_dim(losses)),
              obs = nothing,                                    # [(i₁,j₁), (i₂,j₂), ... (iₒ,jₒ)]
              observed_features = fill(1:size(A,2), size(A,1)), # [1:n, 1:n, ... 1:n] m times
              observed_examples = fill(1:size(A,1), size(A,2)), # [1:m, 1:m, ... 1:m] n times
              offset = false, scale = false,
              checknan = true, sparse_na = true,
              feature_ids=nothing, feature_graphs=nothing, 
              instance_ids=nothing) 
    # Check dimensions of the arguments
    m,n = size(A)
    if length(losses)!=n error("There must be as many losses as there are columns in the data matrix") end
    if length(rx)!=m error("There must be either one X regularizer or as many X regularizers as there are rows in the data matrix") end
    if length(ry)!=n error("There must be either one Y regularizer or as many Y regularizers as there are columns in the data matrix") end
    if size(X)!=(k,m) error("X must be of size (k,m) where m is the number of rows in the data matrix. This is the transpose of the standard notation used in the paper, but it makes for better memory management. \nsize(X) = $(size(X)), size(A) = $(size(A)), k = $k") end
    if size(Y)!=(k,embedding_dim(losses)) error("Y must be of size (k,d) where d is the sum of the embedding dimensions of all the losses. \n(1 for real-valued losses, and the number of categories for categorical losses).") end

    # Determine observed entries of data
    if obs==nothing && sparse_na && isa(A,SparseMatrixCSC)
        obs = findall(!iszero, A) # observed indices (list of CartesianIndices)
    end
    if obs==nothing # if no specified array of tuples, use what was explicitly passed in or the defaults (all)
        glrm = RRGLRM(A,losses,rx,ry,k, observed_features, observed_examples, X,Y, 
                      feature_ids, feature_graphs, instance_ids)
    else # otherwise unpack the tuple list into arrays
        glrm = RRGLRM(A,losses,rx,ry,k, sort_observations(obs,size(A)...)..., X,Y,
                      feature_ids, feature_graphs, instance_ids)
    end

    # check to make sure X is properly oriented
    if size(glrm.X) != (k, size(A,1))
        glrm.X = glrm.X'
    end
    # check none of the observations are NaN
    if checknan
        for i=1:size(A,1)
            for j=glrm.observed_features[i]
                if isnan(A[i,j])
                    error("Observed value in entry ($i, $j) is NaN.")
                end
            end
        end
    end

    if scale # scale losses (and regularizers) so they all have equal variance
        equilibrate_variance!(glrm)
    end
    if offset # don't penalize the offset of the columns
        add_offset!(glrm)
    end
    return glrm
end


function RRGLRM(A, feature_losses::Vector{Loss},
                   feature_ids::Vector, 
                   feature_graphs::Vector{ElUgraph},
                   instance_ids::Vector,
                   instance_groups::Vector;
                   offset=false, scale=false)

    k = length(feature_graphs)

    # Convert the feature graphs into 
    #  (a) a set of graph regularizers and
    #  (b) an extended set of features (observed and latent)
    ry, extended_feature_ids = ugraphs_to_regularizers(feature_graphs)
    extended_losses = extend_losses(feature_losses,
                                    feature_ids,
                                    extended_feature_ids)

    # Convert the instance group hierarchy into  
    #  (a) a set of graph regularizers and
    #  (b) an extended set of instances (observed and latent)
    instance_hierarchy = get_instance_hierarchy(instance_ids, instance_groups)
    rx, extended_instance_ids = hierarchy_to_regularizers(instance_hierarchy, k)

    # We are now ready to assemble the matrix for our
    # factorization problem!
    extended_A = assemble_matrix(A, feature_ids, extended_feature_ids, 
                                    instance_ids, extended_instance_ids)
     
    # Get the observed indices
    obs = findall(!isnan, extended_A)

    rrglrm = RRGLRM(extended_A, extended_losses, rx, ry, k;
                    feature_ids=extended_feature_ids, feature_graphs=feature_graphs, 
                    instance_ids=extended_instance_ids, obs=obs,
                    offset=offset, scale=scale)

    return rrglrm
end


# Reconstruct a model from its factors.
# Sufficient for transforming new instances.
function RRGLRM(feature_factor::Matrix, instance_factor::Matrix, 
                feature_ids::Vector, feature_losses::Vector, 
                instance_ids::Vector)

    k = size(feature_factor, 1)    
    m = size(instance_factor, 2)
    n = size(feature_factor, 2)

    A = DummyArray([m,n])
    instance_reg = fill(ZeroReg(), m)
    feature_reg = fill(ZeroReg(), n)

    println("A: ", size(A))
    println("feature_losses: ", size(feature_losses))
    println("feature_ids: ", size(feature_ids))
    println("instance_ids: ", size(instance_ids))

    rrglrm = RRGLRM(A, feature_losses, instance_reg, feature_reg, k;
                    X=instance_factor, Y=feature_factor,
                    feature_ids=feature_ids, feature_graphs=nothing,
                    instance_ids=instance_ids)

    return rrglrm

end

parameter_estimate(rrglrm::RRGLRM) = (rrglrm.X, rrglrm.Y)


function scale_regularizer!(rrglrm::RRGLRM, newscale::Number)
    mul!(rrglrm.rx, newscale)
    mul!(rrglrm.ry, newscale)
    return rrglrm
end

