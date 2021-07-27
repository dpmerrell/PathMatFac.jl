
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
# Model assembly
#############################################################

function feature_to_loss(feature_name)
    omic_type = split(feature_name, "_")[end]
    if DEFAULT_OMIC_DTYPES[omic_type] <: Bool
        return LogisticLoss()
    else
        return QuadLoss() 
    end
end


function set_matrix_dtypes!(matrix, col_idx, col_names)
    for idx in col_idx
        omic_type = split(col_names[idx], "_")[end]
        col_type = DEFAULT_OMIC_DTYPES[omic_type] 
        
        not_nan = findall(!isnan, matrix[:,idx])
        matrix[not_nan, idx] = convert(Vector{col_type}, matrix[not_nan, idx])
    end
end


function assemble_model(pathways, omic_matrix, feature_names, 
                        sample_ids, sample_groups)

    # Extend the pathways to include "central dogma" entities
    extended_pwys, featuremap = load_pathways(pathways, feature_names)

    omic_types = get_omic_types(feature_names)
    
    # Add data nodes to the pathways
    augmented_pwys = [add_data_nodes_to_pathway(pwy, featuremap, 
                                                omic_types, DEFAULT_OMIC_MAP) 
                      for pwy in extended_pwys] 

    println("AUGMENTED PATHWAYS:")
    println(augmented_pwys)

    srt_feature_names = sort_features(feature_names)
    println("SORTED FEATURE NAMES")
    println(srt_feature_names)

end


function assemble_model_from_sifs(sif_filenames, omic_matrix, feature_names,
                                  sample_ids, sample_groups)
    
    pathways = read_all_sif_files(sif_filenames)
    assemble_model(pathways, omic_matrix, feature_names,
                   sample_ids, sample_groups)

end

#function extend_losses(losses, features, ext_features)
#
#    ext_losses = fill(QuadLoss(), size(ext_features,1))
#    ext_losses = convert(Vector{Loss}, ext_losses)
#
#    feat_to_loss = Dict(feat => loss for (feat, loss) in zip(features, losses))
#
#    for (i, feat) in enumerate(ext_features)
#        if feat in keys(feat_to_loss)
#            ext_losses[i] = feat_to_loss[feat]
#        end
#    end
#
#    return ext_losses
#end

