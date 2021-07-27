
export assemble_model_from_sifs

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

function edgelist_to_spmat(edgelist, name_to_idx; epsilon=1e-5)

    N = length(name_to_idx)


end


#############################################################
# Model assembly
#############################################################

function assemble_model(pathways, omic_matrix, feature_names, 
                        sample_ids, sample_groups)

    # Extend the pathways to include "central dogma" entities
    extended_pwys, featuremap = load_pathways(pathways, feature_names)

    omic_types = get_omic_types(feature_names)
    
    # Add data nodes to the pathways
    augmented_pwys = [add_data_nodes_to_pathway(pwy, featuremap, 
                                                omic_types, DEFAULT_OMIC_MAP) 
                      for pwy in extended_pwys] 

    # Assemble the augmented feature set
    augmented_features = collect(get_all_nodes_many(augmented_pwys))
    println("AUGMENTED PATHWAYS:")
    println(augmented_pwys)
    augmented_features = sort_features(augmented_features)
    println("SORTED FEATURE NAMES")
    println(augmented_features)

    # Assemble the regularizer sparse matrices
    feat_to_idx = value_to_idx(augmented_features)
    feature_reg_mats = edgelists_to_spmats(augmented_pwys, feat_to_idx)

    # - sample regularizer matrices
    #     * build the patient edge list from the "sample groups" vector
    #     * translate the patient edge list to a sparse matrix
    #     * generate k copies of it, total
    #
    # Assemble the vector of losses
    #
    # Initialize the GPUMatFac model
    #
    # Assemble the data matrix
    #

end


function assemble_model_from_sifs(sif_filenames, omic_matrix, feature_names,
                                  sample_ids, sample_groups)
    
    pathways = read_all_sif_files(sif_filenames)
    assemble_model(pathways, omic_matrix, feature_names,
                   sample_ids, sample_groups)

end


