
using JSON
using LinearAlgebra


function pwy_to_matrix(edges)

    idxs = Set()
    for edge in edges
        push!(idxs, edge[1]+1)
        push!(idxs, edge[2]+1)
    end
    encoder = Dict([idx => i for (i, idx) in enumerate(idxs)])
    dim = length(idxs)
    
    result = zeros(dim,dim)
    for edge in edges
        result[encoder[edge[1]], encoder[edge[2]]] = edge[3]
    end
   
    decoder = zeros(Int64, dim)
    for (idx, i) in encoder
        decoder[i] = idx
    end 
    decoder = Dict([i => idx for (idx, i) in encoder])
    
    return result, decoder

end


function compute_dyn_eigenvectors(pwy, n_entities, epsilon=1e-10)

    mat, row_decoder = pwy_to_matrix(pwy)

    emat = exp(mat)
    F = eigen(mat)
    stationary_inds = [i for (i, v) in enumerate(F.values) if abs(v-1.0) < epsilon]

    enc_vecs = F.vectors[:,stationary_inds]

    result = zeros(n_entities, size(enc_vecs,2))

    for col=1:size(result,2)
        result[row_decoder,col] .= enc_vecs[:,col]
    end

    return result
end


function compute_edge_sum(pwy, n_entities)
    sums = zeros(n_entities, 1)
    for edge in pwy
        sums[edge[1]+1] += edge[3]
        sums[edge[2]+1] += edge[3]
    end

    return sums 
end


function compute_bag_of_entities(pwy, n_entities)
    pwy_entities = Set()
    for edge in pwy
        push!(pwy_entities, pwy[1])
        push!(pwy_entities, pwy[2])
    end

    bag = [ Float64(i-1 in pwy_entities) for i=1:n_entities]

    return reshape(bag, (n_entities, 1))
end


"""
Convert an array of pathways (represented as edge lists)
into a matrix whose columns represent those pathways.

Returns the matrix, along with a `col_decoder` vector 
that maps matrix columns to pathways.

(In general there may be multiple columns per pathway.)
"""
function pathways_to_vectors(entity_list, pwy_list, 
                             vectorize="edge_sums")

    n_entities = length(entity_list)
    
    vecs = []
    col_decoder = []

    n_vecs = 0

    if vectorize == "edge_sums"
        vec_method = compute_edge_sum
    elseif vectorize == "bag_of_entities"
        vec_method = compute_bag_of_entities
    elseif vectorize == "dyn_eigenvectors"
        vec_method = compute_dyn_eigenvectors
    else
        throw(DomainError())

    col_idx = 1
    for (i, pwy) in enumerate(pwy_list)
        pwy_vecs = vec_method(pwy, n_entities)
        push!(vecs, pwy_vecs)
        new_col_idx = col_idx+size(pwy_vecs,2) 
        col_decoder[col_idx:new_col_idx] .= i
        col_idx = new_col_idx
    end

    return hcat(vecs...), col_decoder

end



