
using CSV, DataFrames, HDF5

export load_pathways, load_pathway_sifs, populate_featuremap, 
       get_omic_feature_names, get_omic_instances, get_omic_groups


#############################################################
# Omic data IO
#############################################################

function get_omic_feature_names(omic_hdf)

    idx = h5open(omic_hdf, "r") do file
        read(file, "features")
    end

    return idx 
end


function get_omic_instances(omic_hdf)

    patients = h5open(omic_hdf, "r") do file
        read(file, "instances")
    end

    return patients 
end


function get_omic_groups(omic_hdf)

    cancer_types = h5open(omic_hdf, "r") do file
        read(file, "cancer_types")
    end

    return cancer_types
end


function get_omic_data(omic_hdf)

    dataset = h5open(omic_hdf, "r") do file
        read(file, "data")
    end

    # Julia reads arrays from HDF files
    # in the (weird) FORTRAN order
    return permutedims(dataset)
end


function build_instance_hierarchy(instance_ids::Vector{T}, 
                                  instance_groups::Vector{String}) where T

    hierarchy = Dict(gp => T[] for gp in unique(instance_groups))

    for (i, inst) in enumerate(instance_ids)
        push!(hierarchy[instance_groups[i]], inst)
    end

    return hierarchy

end


"""
    Given a nested dictionary of patients,
    generate an undirected graph representation
    of that tree; and then create a sparse matrix
    for it.
"""
function hierarchy_to_matrix(patient_hierarchy)

    edges = []
    all_nodes = []

    # Recursively build an edge list 
    # from the dictionary
    function rec_h2m(parent_name, children)
        push!(all_nodes, parent_name)
        if typeof(children) <: Vector
            for child_name in children
                push!(edges, [parent_name, child_name, 1.0])
                push!(all_nodes, child_name)
            end
        elseif typeof(children) <: Dict
            for child_name in sort(collect(keys(children)))
                push!(edges, [parent_name, child_name, 1.0])
                rec_h2m(child_name, children[child_name])
            end
        end
    end
    rec_h2m("", patient_hierarchy) 
    graph = construct_elugraph(edges)

    node_to_idx = value_to_idx(all_nodes) # Dict(v => idx for (idx,v) in enumerate(all_nodes))

    matrix = ugraph_to_matrix(graph, node_to_idx) 

    return (matrix, all_nodes)

end


######################################################
# Pathway IO
######################################################

"""
    read a SIF file, returning it as a 
    vector of vectors
"""
function read_sif_file(sif_file::String)
    df = DataFrame!(CSV.File(sif_file))
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
function extend_pathway(pathway)

    proteins = Set{String}()
    new_edges = [] 

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



#
#
#function ugraph_to_matrix(graph::ElUgraph, node_to_idx, 
#                          epsilon=nothing, 
#                          standardize=true)
#    N = length(node_to_idx)
#
#    if epsilon == nothing
#        epsilon = 1.0 / sqrt(N)
#    end
#
#    I = Int64[]
#    J = Int64[]
#    V = Float64[]
#    diag_entries = fill(epsilon, N)
#
#    for edge in graph.edges
#        # Append off-diagonal entries
#        push!(I, node_to_idx[edge[1]])
#        push!(J, node_to_idx[edge[2]])
#        push!(V, -1.0*edge[3])
#
#        # Increment diagonal entries
#        # (maintain diagonal dominance)
#        diag_entries[node_to_idx[edge[1]]] += 1.0
#    end
#
#    # Now append diagonal entries
#    for (i,v) in enumerate(diag_entries)
#        push!(I, i)
#        push!(J, i)
#        push!(V, v)
#    end 
#
#    m = sparse(I, J, V, N, N)
#   
#    # optionally: rescale matrix s.t. we
#    # have ones on the diagonal (i.e., the 
#    # matrix is an inverted correlation matrix) 
#    if standardize
#        standardizer = sparse(1:N, 1:N, 1.0./sqrt.(diag_entries), N, N)
#        m = standardizer * m * standardizer 
#    end
#
#    return m 
#end
#
#
#function ugraphs_to_matrices(ugraphs::Vector{ElUgraph})
#
#    # Get all of the nodes from all of the graphs
#    all_nodes = Set{String}()
#    for graph in ugraphs
#        for edge in graph.edges
#            push!(all_nodes, edge[1])
#        end
#    end
#
#    # in lexicographic order
#    all_nodes = sort(collect(all_nodes))
#    N = size(all_nodes,1)
#
#    # create a mapping from node names to indices
#    node_to_idx = value_to_idx(all_nodes) # Dict(v => i for (i, v) in enumerate(all_nodes))
#
#    # now translate the ugraphs to sparse matrices
#    matrices = [ugraph_to_matrix(u, node_to_idx) for u in ugraphs]
#
#    return matrices, all_nodes
#end
#
#
#function pwy_to_matrix(pwy, featuremap, data_type_map;
#                       extension_strategy="sparse_latent")
#
#    ext_pwy = add_data_nodes_to_pathway(pwy, featuremap,
#                                             data_type_map;
#                                             strategy=extension_strategy)
#    
#
#    # Get all of the nodes from all of the graphs
#    all_nodes = Set{String}()
#    for graph in ugraphs
#        for edge in graph.edges
#            push!(all_nodes, edge[1])
#        end
#    end
#
#    # in lexicographic order
#    all_nodes = sort(collect(all_nodes))
#    N = size(all_nodes,1)
#
#    # create a mapping from node names to indices
#    node_to_idx = value_to_idx(all_nodes) # Dict(v => i for (i, v) in enumerate(all_nodes))
#
#    # now translate the ugraphs to sparse matrices
#    matrices = [ugraph_to_matrix(u, node_to_idx) for u in ugraphs]
#
#    return matrices, all_nodes
#end
#
#
#"""
#    Given a vector of pathways and a populated featuremap,
#    return a corresponding dictionary of sparse matrices
#    and an array that maps indices to pathway nodes
#"""
#function pathways_to_ugraphs(pathways, featuremap;
#                             data_types=DEFAULT_OMICS, 
#                             data_type_map=DEFAULT_OMIC_MAP,
#                             pwy_data_augmentation="sparse_latent",
#                             pwy_to_ugraph="symmetrize")
#
#    # Augment the pathways with additional nodes
#    # to represent the data    
#    ext_pathways = [add_data_nodes_to_pathway(pwy, featuremap, data_types, data_type_map;
#                                              strategy=pwy_data_augmentation)
#                                              for pwy in pathways]
#
#    # Pathways are currently interpreted as
#    # directed graphs. Convert to undirected graphs
#    ugraphs = ElUgraph[construct_elugraph(pwy) for pwy in ext_pathways]
#
#    return ugraphs 
#end
#
#
#function ugraphs_to_regularizers(ugraphs::Vector{ElUgraph}; offset=false)
#
#    matrices, ext_features = ugraphs_to_matrices(ugraphs)
#
#    regularizers = matrices_to_regularizers(matrices, ext_features; offset=offset)
#
#    return regularizers, ext_features
#end


function get_all_proteins(pathways)

    proteins = Set{String}()
    for edge_list in pathways
        for edge in edge_list
            for node in edge[1:2]
                tok = split(node,"_")
                if tok[2] == "protein"
                    push!(proteins, tok[1])
                end
            end
        end
    end

    return proteins
end


function get_all_nodes(edge_list)
    nodes = Set{String}()
    for edge in edge_list
        push!(nodes, edge[1])
        push!(nodes, edge[2])
    end
    return nodes
end


function get_all_nodes_many(edge_lists)
    nodes = Set{String}()
    for edge_list in edge_lists
        union!(nodes, get_all_nodes(edge_list))
    end
    return nodes
end


function get_omic_types(feature_names)
    return collect(Set([split(feat, "_")[end] for feat in feature_names]))
end


################################################################
# Mapping features to pathways
################################################################

"""
    Prepare a data structure that will map
    graph nodes to rows of data.
"""
function initialize_featuremap(all_pwy_proteins, all_data_types)

    result = Dict( string(pro, "_", dt) =>  Vector{Int}() 
                                   for dt in all_data_types  
                                       for pro in all_pwy_proteins)
    result["unmapped"] = Vector{Int}()
    return result
end


"""
    Given an empty featuremap, populate it from the array 
    of features. 
"""
function populate_featuremap(featuremap, features)

    for (idx, feat) in enumerate(features)
        
        tok = split(feat, "_")
        # extract the protein names
        prot_names = split(tok[1], " ")
        
        omic_datatype = tok[end]
 
        # for each protein name
        for protein in prot_names
            k = string(protein, "_", omic_datatype)
            if k in keys(featuremap)
                push!(featuremap[k], idx)
            end
        end
    end

    return featuremap
end



####################################################
# Putting it all together...
####################################################

function load_pathways(pwy_vec, feature_names)

    extended_pwys = [extend_pathway(pwy) for pwy in pwy_vec]
    all_proteins = get_all_proteins(extended_pwys)
    data_kinds = get_omic_types(feature_names)
    empty_feature_map = initialize_featuremap(all_proteins, data_kinds)

    populated_feature_map = populate_featuremap(empty_feature_map, feature_names)


    return (extended_pwys, populated_feature_map)
end


function load_pathway_sifs(sif_filenames, feature_names)

    pathways = read_all_sif_files(sif_filenames)
    extended_pwys, feature_map = load_pathways(pathways, feature_names)
 
    return (extended_pwys, feature_map)
end 


