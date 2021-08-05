
using SparseArrays

export assemble_model, assemble_model_from_sifs

##############################################################
# Add data nodes to the pathways
##############################################################

function add_data_nodes_sparse_latent(pathway,
                                      feature_map::Dict, 
                                      data_types, 
                                      data_node_map::Dict)

    # Get all of the proteins from this pathway
    proteins = get_all_proteins([pathway])

    # for each protein, add edges from data nodes to
    # to the correct parts of the graph
    for protein in proteins
        for dt in data_types
            for idx in feature_map[string(protein, "_", dt)]
                data_node = string(protein, "_", dt) 
                v = data_node_map[dt]            
                push!(pathway, [string(protein, "_", v[1]), data_node, v[2]])
            end
        end
    end

    return pathway
end


function add_data_nodes_to_pathway(pathway, featuremap, data_types, data_type_map;
                                   strategy="sparse_latent")
    
    if strategy == "sparse_latent"
        pwy = add_data_nodes_sparse_latent(pathway, featuremap, data_types, data_type_map)
    else
        throw(DomainError(strategy, "not a valid option for `strategy`"))
    end

    return pwy

end


#############################################################
# Assembling sparse matrices
#############################################################

function edgelist_to_spmat(edgelist, node_to_idx; epsilon=1e-5)

    N = length(node_to_idx)

    # make safe against redundancies.
    # in case of redundancy, keep the latest
    edge_dict = Dict()
    for e in edgelist
        e1 = node_to_idx[e[1]]
        e2 = node_to_idx[e[2]]
        u = max(e1, e2)
        v = min(e1, e2)
        edge_dict[(u,v)] = e[3]
    end

    I = Int64[] 
    J = Int64[] 
    V = Float64[] 
    diagonal = fill(epsilon, N)

    # Off-diagonal entries
    for (idx, value) in edge_dict
        # below the diagonal
        push!(I, idx[1])
        push!(J, idx[2])
        push!(V, value)
        
        # above the diagonal
        push!(I, idx[2])
        push!(J, idx[1])
        push!(V, value)

        # increment diagonal entries
        # (maintain positive definite-ness)
        av = abs(value)
        diagonal[idx[1]] += av
        diagonal[idx[2]] += av
    end

    # diagonal entries
    for i=1:N
        push!(I, i)
        push!(J, i)
        push!(V, diagonal[i])
    end

    result = sparse(I, J, V)

    return result
end

function edgelists_to_spmats(edgelists, node_to_idx)
    return [edgelist_to_spmat(el, node_to_idx) for el in edgelists]
end

#############################################################
# Patient groups
#############################################################

function augment_samples(sample_ids, group_ids; rooted=true)
    result = vcat(sample_ids, unique(group_ids))
    if rooted
        push!(result, "ROOT")
    end
    return result
end

function create_sample_edgelist(sample_id_vec, group_vec; rooted=true)
    
    m = length(sample_id_vec)
    @assert m == length(group_vec)

    edgelist = []

    # Tie samples to their groups
    for i=1:m
        push!(edgelist, [group_vec[i], sample_id_vec[i], 1])
    end

    # If we're rooted, then tie the groups to a root node
    if rooted
        for gp in unique(group_vec)
            push!(edgelist, ["ROOT", gp, 1])
        end
    end

    return edgelist
end


#############################################################
# Model assembly
#############################################################

function assemble_model(pathways, omic_matrix, feature_names, 
                        sample_ids, sample_groups;
                        rooted_samples=true)

    K = length(pathways)

    # Extend the pathways to include "central dogma" entities
    extended_pwys, featuremap = load_pathways(pathways, feature_names)

    omic_types = get_omic_types(feature_names)
    
    # Add data nodes to the pathways
    augmented_pwys = [add_data_nodes_to_pathway(pwy, featuremap, 
                                                omic_types, DEFAULT_OMIC_MAP) 
                      for pwy in extended_pwys] 

    # Assemble the augmented feature set
    augmented_features = collect(get_all_nodes_many(augmented_pwys))
    augmented_features = sort_features(augmented_features)

    # Assemble the regularizer sparse matrices
    feat_to_idx = value_to_idx(augmented_features)
    feature_reg_mats = edgelists_to_spmats(augmented_pwys, feat_to_idx)
    #println("FEATURE REGULARIZATION MATRICES:")
    #for mat in feature_reg_mats
    #    println(mat)
    #end

    # build the sample edge list from the "sample groups" vector
    augmented_samples = augment_samples(sample_ids, sample_groups, rooted=rooted_samples) 
    sample_edgelist = create_sample_edgelist(sample_ids, sample_groups, rooted=rooted_samples)
    sample_to_idx = value_to_idx(augmented_samples)

    # translate the sample edge list to a sparse matrix
    sample_reg_mats = fill(edgelist_to_spmat(sample_edgelist, sample_to_idx), K)
    #println("SAMPLE REGULARIZATION MATRIX:")
    #println(sample_reg_mats[1])
    
    # Assemble the vector of losses
    loss_vector = Loss[get_loss(feat)(1.0) for feat in augmented_features]
    #println("LOSS VECTOR")
    #println(loss_vector)

    # Initialize the GPUMatFac model
    matfac_model = MatFacModel(sample_reg_mats, feature_reg_mats, loss_vector)

    return MultiomicModel(matfac_model, augmented_features, augmented_pwys,
                          feat_to_idx, augmented_samples, sample_edgelist,
                          sample_to_idx, omic_matrix)

end


function assemble_model_from_sifs(sif_filenames, omic_matrix, feature_names,
                                  sample_ids, sample_groups)
    
    pathways = read_all_sif_files(sif_filenames)
    assemble_model(pathways, omic_matrix, feature_names,
                   sample_ids, sample_groups)

end


