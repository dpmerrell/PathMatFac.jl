

export assemble_model, assemble_model_from_sifs

##############################################################
# Add data nodes to the pathways
##############################################################

function add_data_nodes_sparse_latent(pathway,
                                      feature_map::Dict, 
                                      unique_assays, 
                                      assay_map::Dict)

    # Get all of the proteins from this pathway
    proteins = get_all_proteins([pathway])

    # for each protein, add edges from data nodes to
    # to the correct parts of the graph
    for protein in proteins
        for assay in unique_assays
            for idx in feature_map[(protein, assay)]
                data_node = (protein, assay) 
                v = assay_map[assay]
                pwy_node = (string(protein, "_", v[1]), "")
                push!(pathway, [pwy_node, data_node, v[2]])
            end
        end
    end

    return pathway
end


function add_data_nodes_to_pathway(pathway, featuremap, unique_assays, assay_map;
                                   strategy="sparse_latent")
    
    if strategy == "sparse_latent"
        pwy = add_data_nodes_sparse_latent(pathway, featuremap, 
                                           unique_assays, assay_map)
    else
        throw(DomainError(strategy, "not a valid option for `strategy`"))
    end

    return pwy

end


#############################################################
# Assembling sparse matrices
#############################################################

function edgelist_to_spmat(edgelist, node_to_idx; epsilon=1e-5, verbose=false)

    N = length(node_to_idx)

    # make safe against redundancies.
    # in case of redundancy, keep the latest
    edge_dict = Dict()
    for edge in edgelist
        if verbose
            println(edge)
        end
        e1 = node_to_idx[edge[1]]
        e2 = node_to_idx[edge[2]]
        u = max(e1, e2)
        v = min(e1, e2)
        edge_dict[(u,v)] = edge[3]
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
        push!(V, -value)
        
        # above the diagonal
        push!(I, idx[2])
        push!(J, idx[1])
        push!(V, -value)

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

function edgelists_to_spmats(edgelists, node_to_idx; verbose=false)
    return [edgelist_to_spmat(el, node_to_idx; verbose=verbose) for el in edgelists]
end

#############################################################
# Patient groups
#############################################################

function augment_samples(sample_ids, group_ids; rooted=false)
    result = vcat(sample_ids, unique(group_ids))
    if rooted
        push!(result, "ROOT")
    end
    return result
end

function create_sample_edgelist(sample_id_vec, group_vec; rooted=false)
    
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
#
#
##############################################################
## Assemble the data matrix
##############################################################
#
#function augment_omic_matrix(omic_matrix, original_features, augmented_features, 
#                                          original_samples, augmented_samples)
#
#    feat_idx_vec, aug_feat_idx_vec = keymatch(original_features, augmented_features)
#    sample_idx_vec, aug_sample_idx_vec = keymatch(original_samples, augmented_samples)
#
#    M = length(augmented_samples)
#    N = length(augmented_features)
#    result = fill(NaN, M, N) 
#
#    for (f_idx, aug_f_idx) in zip(feat_idx_vec, aug_feat_idx_vec)
#        result[aug_sample_idx_vec, aug_f_idx] .= omic_matrix[sample_idx_vec, f_idx] 
#    end
#
#    return result
#end
#
#
##############################################################
## Model assembly
##############################################################
#

function construct_assay_edgelist(features, assays)
    
    unique_assay_set = Set(assays)
    edgelist = Vector{Any}[]

    for feat in features
        if feat[2] in unique_assay_set
            push!(edgelist, [(feat[2],""), feat, 1])
        end
    end

    return edgelist
end


function prep_pathways(pathway_sif_data, feature_genes, feature_assays;
                       assay_map=DEFAULT_ASSAY_MAP)


    # Extend the pathways to include "central dogma" entities
    extended_pwys, featuremap = load_pathways(pathway_sif_data, 
                                              feature_genes, 
                                              feature_assays)

    unique_assays = unique(feature_assays)

    # Add data nodes to the pathways
    augmented_pwys = [add_data_nodes_to_pathway(pwy, featuremap, 
                                                unique_assays, 
                                                assay_map) 
                      for pwy in extended_pwys] 

    # Assemble the augmented feature set
    augmented_features = collect(get_all_nodes_many(augmented_pwys))
    
    return augmented_pwys, augmented_features
end


function add_assay_nodes(prepped_features, feature_assays)
    
    unique_assays = unique(feature_assays)
    assay_nodes = [(assay,"") for assay in unique_assays]
    prepped_features = vcat(prepped_features, assay_nodes) 
    return prepped_features
end


function assemble_feature_reg_mats(pathway_sif_data, feature_genes, feature_assays;
                                   assay_map=DEFAULT_ASSAY_MAP)

    prepped_pathways, prepped_features = prep_pathways(pathway_sif_data,
                                                       feature_genes,
                                                       feature_assays)
    
    # Need to add additional virtual nodes for
    # assay-wise regularization
    prepped_features = add_assay_nodes(prepped_features, feature_assays)

    # Sort features by loss, assay, and gene
    prepped_features = sort_features(prepped_features)
    feat_to_idx = value_to_idx(prepped_features)
    
    # Assemble the regularizer sparse matrices
    feature_reg_mats = edgelists_to_spmats(prepped_pwys, feat_to_idx)
    
    # Create a graph connecting features of the same assay 
    assay_edgelist = construct_assay_edgelist(prepped_features, feature_assays)
    assay_reg_mat = edgelist_to_spmat(assay_edgelist, feat_to_idx)


    return feature_reg_mats, assay_reg_mat, prepped_features, feat_to_idx
end


function assemble_instance_reg_mat(sample_ids, sample_groups)
    
    # build the sample edge list from the "sample groups" vector
    augmented_samples = augment_samples(sample_ids, sample_groups) 
    sample_edgelist = create_sample_edgelist(sample_ids, sample_groups)
    aug_sample_to_idx = value_to_idx(augmented_samples)

    # translate the sample edge list to a sparse matrix
    sample_reg_mat = edgelist_to_spmat(sample_edgelist, aug_sample_to_idx)

    return sample_reg_mat, augmented_samples, aug_sample_to_idx
end


function assemble_model(pathway_sif_data,  
                        sample_ids, sample_groups,
                        feature_genes, feature_assays)

    # Construct the sample regularizer matrix (for X)
    sample_reg_mat, 
    internal_samples, 
    internal_sample_to_idx = assemble_instance_reg_mat(sample_ids, 
                                                       sample_groups)


    # Construct the pathway-based feature regularizer matrices (for Y)
    # and the assay-based regularizer matrix (for mu, sigma)
    feature_reg_mats, 
    assay_reg_mat, 
    augmented_features, 
    aug_feat_to_idx = assemble_feature_reg_mats(pathway_sif_data, 
                                                feature_genes, 
                                                feature_assays)


    # Construct MatFacModel
    # TODO: still need sample_groups, feature_groups, feature_loss_names
    matfac = MatFacModel(sample_reg_mat, feature_reg_mats, 
                         assay_reg_mat, assay_reg_mat,
                         sample_groups, feature_groups,
                         feature_loss_names)

    # TODO: still need most of these things
    MultiomicModel(matfac, sample_ids, internal_sample_idx,
                   aug_sample_ids, feature_genes, feature_assays,
                   internal_feature_idx, aug_feature_genes,
                   aug_feature_assays)


end


function assemble_model_from_sifs(sif_filenames, omic_matrix, 
                                  feature_genes, feature_assays,
                                  sample_ids, sample_groups)
    
    pathway_sif_data = read_all_sif_files(sif_filenames)
    assemble_model(pathway_sif_data, omic_matrix,
                   feature_genes, feature_assays,
                   sample_ids, sample_groups)

end


