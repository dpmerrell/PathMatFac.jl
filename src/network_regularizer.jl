

mutable struct NetworkRegularizer

    AA::Tuple # Tuple of K matrices encoding relationships 
              # beween *observed* features

    AB::Tuple # Tuple of K matrices encoding relationships
              # between *observed and unobserved* features

    BB::Tuple # Tuple of K matrices encoding relationships
              # between *unobserved* features

    B_matrix::AbstractMatrix # a (K x N_unobserved) matrix
                             # containing estimates of 
                             # unobserved quantities
end

@functor NetworkRegularizer

function NetworkRegularizer(edgelists; observed=nothing,
                                       weight=1.0)

    # Collect all the nodes from the edgelists
    all_nodes = Set()
    for el in edgelists
        for edge in el
            push!(all_nodes, edge[1])
            push!(all_nodes, edge[2])
        end
    end

    # Figure out which nodes are observed,
    # and which are unobserved
    if observed == nothing
        observed = sort(collect(all_nodes))
        unobserved = []
    else
        # We'll just order these lexicographically
        unobserved = sort(collect(setdiff(all_nodes,observed)))
    end
    n_obs = length(observed)
    n_unobs = length(unobserved)

    # For constructing the sparse matrices, we simply
    # concatenate the observed and unobserved nodes
    allnodes_sorted = vcat(observed, unobserved)

    node_to_idx = value_to_idx(allnodes_sorted) 
    spmats = edgelists_to_spmats(edgelists, node_to_idx)

    # Rescale the sparse matrices with the 
    # regularization weight
    for spmat in spmats
        rescale!(spmat, weight)
    end
   
    AA = Tuple(map(mat->mat[1:n_obs,1:n_obs], spmats))
    AB = Tuple(map(mat->mat[1:n_obs,n_obs+1:end], spmats))
    BB = Tuple(map(mat->mat[n_obs+1:end, n_obs+1:end], spmats))

    K = length(edgelists)
    B_matrix = randn(K, n_unobs)

    return NetworkRegularizer(AA, AB, BB, B_matrix)
end


quadratic(u::AbstractVector, 
          X::AbstractMatrix, 
          v::AbstractVector) = dot(u, (X*v))

quadratic(X::AbstractMatrix, v::AbstractVector) = dot(v, (X*v))


#################################################
# Matrix row-regularizer
#################################################

function (nr::NetworkRegularizer)(X::AbstractMatrix)

    loss = 0.0
    K = size(X,1)
    for k=1:K
        loss += quadratic(nr.AA[k], X[k,:])
        loss += 2*quadratic(X[k,:], nr.AB[k], nr.B_matrix[k,:])
        loss += quadratic(nr.BB[k], nr.B_matrix[k,:])
    end
    return 0.5*loss
end


function ChainRules.rrule(nr::NetworkRegularizer, X::AbstractMatrix)

    K, MA = size(X)
    _, MB = size(nr.B_matrix)

    # Pre-allocate these matrix-vector products
    xAA = similar(X, K,MA)
    xAB = similar(X, K,MB)
    ABu = similar(nr.B_matrix, MA,K)
    BBu = similar(nr.B_matrix, MB,K)

    loss = 0.0
    for k=1:K
        # Parts of the gradients
        xAA[k,:] = transpose(X[k,:])*nr.AA[k]
        xAB[k,:] = transpose(X[k,:])*nr.AB[k]
        ABu[:,k] = nr.AB[k]*nr.B_matrix[k,:]
        BBu[:,k] = nr.BB[k]*nr.B_matrix[k,:]

        # Full loss computation
        loss += 0.5*dot(xAA, X[k,:])
        loss += dot(X[k,:], ABu[:,k])
        loss += 0.5*dot(nr.B_matrix[k,:], BBu[:,k])
    end

    function netreg_mat_pullback(loss_bar)

        B_bar = transpose(ABu .+ BBu)
        nr_bar = Tangent{NetworkRegularizer}(B_matrix=B_bar)

        return nr_bar, xAA .+ xAB
    end

    return loss, netreg_mat_pullback
end


#################################################
# Vector regularizer
#################################################

function (nr::NetworkRegularizer)(x::AbstractVector)

    loss = 0.0
    loss += quadratic(nr.AA[1], x)
    loss += 2*quadratic(x, nr.AB[1], nr.B_matrix[1,:])
    loss += quadratic(nr.BB[1], nr.B_matrix[1,:])
    return 0.5*loss
end


function ChainRules.rrule(nr::NetworkRegularizer, x::AbstractVector)
    loss = 0.0

    AAx = nr.AA[1]*x
    xAB = transpose(x)*nr.AB[1]
    ABu = nr.AB[1]*nr.B_matrix[1,:]
    uBB = nr.B_matrix*nr.BB[1]

    loss = 0.5*(dot(x, AAx) + 2*dot(x, ABu) + uBB*nr.B_matrix[1,:])

    function netreg_vec_pullback(loss_bar)

        b_matrix_bar = xAB + uBB
        x_bar = AAx + ABu 

        nr_bar = Tangent{NetworkRegularizer}(B_matrix=b_matrix_bar)

        return nr_bar, x_bar
    end
    
    return loss, netreg_vec_pullback
end



