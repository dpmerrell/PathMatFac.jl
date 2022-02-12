


######################################################
# Pathway IO
######################################################

"""
    read a SIF file, returning it as a 
    vector of vectors
"""
function read_sif_file(sif_file::String)
    df = DataFrame!(CSV.File(sif_file; header=0))
    return [ collect(df[i,:]) for i=1:size(df,1) ]
end


"""
    read many SIF files, returning the result as a 
    vector of vectors of vectors
"""
function read_all_sif_files(sif_files::Vector{String})
    return [read_sif_file(sp) for sp in sif_files]
end


"""
    For each protein in a pathway (vector of vectors),
    extend the pathway to include dna and mrna as suggested
    by the "central dogma". 
"""
function extend_pathway(pathway::Vector)

    proteins = Set{String}()
    new_edges = Vector{Any}[] 

    # Modify the edges to/from proteins
    for edge in pathway 
        u = edge[1]
        
        ent_types = [PWY_SIF_CODE[edge[2][[1]]], 
                     PWY_SIF_CODE[edge[2][[2]]]
                    ]
        target = edge[2][[3]]
        sgn = edge[2][[4]] 
        
        v = edge[3]

        # If u is a protein, replace
        # it with an "activation" node
        if ent_types[1] == "protein"
            new_u = string(u, "_activation")
            push!(proteins, u)
        else
            new_u = string(u, "_", ent_types[1])
        end  

        # If v is a protein, check whether
        # its transcription or activation is
        # targeted by this edge
        if ent_types[2] == "protein"
            push!(proteins, v)
            if target == "transcription"
                new_v = string(v, "_mrna")
            else
                new_v = string(v, "_activation")
            end
        else
            new_v = string(v, "_", ent_types[2])
        end

        # Extract the promotes/suppresses information
        if sgn == ">"
            sgn = 1
        elseif sgn == "|"
            sgn = -1
        end

        push!(new_edges, [new_u, new_v, sgn])
    end
    
    # Add edges to/from the new entities
    # (dna, mrna, protein, activation)
   
    for u in proteins
        push!(new_edges, [string(u, "_dna"), string(u, "_mrna"), 1])
        push!(new_edges, [string(u, "_mrna"), string(u, "_protein"), 1])
        push!(new_edges, [string(u, "_protein"), string(u, "_activation"), 1])
    end

    return new_edges
end


"""
Tag the nodes of the extended pathway with empty strings,
indicating that they will be virtual/unobserved features
in the model.
"""
function tag_pathway(pathway)

    result = Vector{Any}[]
    for edge in pathway
        new_edge = [(edge[1],""), (edge[2],""), edge[3]]
        push!(result, new_edge)
    end

    return result
end


function get_all_proteins(pathways)

    proteins = Set{String}()
    for edge_list in pathways
        for edge in edge_list
            for node in edge[1:2]
                name = node[1]
                tok = split(name,"_")
                if tok[2] == "protein"
                    push!(proteins, tok[1])
                end
            end
        end
    end

    return proteins
end


function get_all_nodes(edge_list)
    nodes = Set()
    for edge in edge_list
        push!(nodes, edge[1])
        push!(nodes, edge[2])
    end
    return nodes
end


function get_all_nodes_many(edge_lists)
    nodes = Set()
    for edge_list in edge_lists
        union!(nodes, get_all_nodes(edge_list))
    end
    return nodes
end



################################################################
# Mapping features to pathways
################################################################

"""
    Prepare a data structure that will map
    graph nodes to rows of data.
"""
function initialize_featuremap(all_pwy_proteins, unique_assays)

    result = Dict( (pro, assay) =>  Int[] 
                                    for assay in unique_assays  
                                        for pro in all_pwy_proteins)
    #result["unmapped"] = Vector{Int}()
    return result
end


"""
    Given an empty featuremap, populate it from the array 
    of features. 
"""
function populate_featuremap(featuremap, feature_genes, feature_assays)

    for (idx, (gene, assay)) in enumerate(zip(feature_genes, feature_assays))
        
        # extract gene names (there may be more than one)
        gene_names = split(gene, " ")
        
        # for each gene name
        for gene in gene_names
            k = (gene, assay)
            if k in keys(featuremap)
                push!(featuremap[k], idx)
            end
        end
    end

    return featuremap
end


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

#######################################################
# ASSAY GRAPH
#######################################################

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


function add_assay_nodes(prepped_features, feature_assays)
    
    unique_assays = unique(feature_assays)
    assay_nodes = [(assay,"") for assay in unique_assays]
    prepped_features = vcat(prepped_features, assay_nodes) 
    return prepped_features
end



####################################################
# Putting it all together...
####################################################



function load_pathways(pwy_vec::Vector{Vector{T}} where T, feature_genes, feature_assays)

    extended_pwys = [extend_pathway(pwy) for pwy in pwy_vec]
    tagged_pwys = [tag_pathway(pwy) for pwy in extended_pwys]
    pwy_proteins = get_all_proteins(tagged_pwys)

    unique_assays = unique(feature_assays)
    empty_feature_map = initialize_featuremap(pwy_proteins, unique_assays)

    populated_feature_map = populate_featuremap(empty_feature_map, 
                                                feature_genes, 
                                                feature_assays)

    return tagged_pwys, populated_feature_map
end


function load_pathways(sif_filenames::Vector{String}, feature_genes, feature_assays)

    pathways = read_all_sif_files(sif_filenames)
    extended_pwys, feature_map = load_pathways(pathways, 
                                               feature_genes,
                                               feature_assays)
 
    return extended_pwys, feature_map
end 


function prep_pathways(pathway_data, feature_genes, feature_assays;
                       assay_map=DEFAULT_ASSAY_MAP)


    # Extend the pathways to include "central dogma" entities
    extended_pwys, featuremap = load_pathways(pathway_data, 
                                              feature_genes, 
                                              feature_assays)

    unique_assays = unique(feature_assays)

    # Add data nodes to the pathways
    augmented_pwys = Vector{Any}[add_data_nodes_to_pathway(pwy, featuremap, 
                                                           unique_assays, 
                                                           assay_map) 
                      for pwy in extended_pwys] 

    # Assemble the augmented feature set
    augmented_features = collect(get_all_nodes_many(augmented_pwys))
    
    return augmented_pwys, augmented_features
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
    feature_reg_mats = edgelists_to_spmats(prepped_pathways, feat_to_idx)
    
    # Create a graph connecting features of the same assay 
    assay_edgelist = construct_assay_edgelist(prepped_features, feature_assays)
    assay_reg_mat = edgelist_to_spmat(assay_edgelist, feat_to_idx)

    return feature_reg_mats, assay_reg_mat, prepped_features, feat_to_idx
end


