
export prep_pathway_graphs, prep_pathway_featuresets 


######################################################
# Pathway SIF file input
######################################################

"""
    read a SIF file, returning it as a 
    vector of vectors
"""
function read_sif_file(sif_file::String)
    df = DataFrame(CSV.File(sif_file; header=0))
    return [ collect(df[i,:]) for i=1:size(df,1) ]
end


"""
    read many SIF files, returning the result as a 
    vector of vectors of vectors
"""
function read_all_sif_files(sif_files::Vector{String})
    return [read_sif_file(sp) for sp in sif_files]
end

######################################################
# Construct the edgelists
######################################################

"""
    Convert a SIF pathway description to a
    list of edges in a more useful format.
"""
function sif_to_edgelist(pwy_sif::Vector)

    new_edges = Vector{Any}[] 

    # Extract sign and target attributes
    # from the interaction
    for edge in pwy_sif 
        
        u = edge[1]
        source = PWY_SIF_CODE["a"]
        target = PWY_SIF_CODE[edge[2][[1]]]
        sgn = PWY_SIF_CODE[edge[2][[2]]]

        v = edge[3]

        push!(new_edges, [string(u, "_", source), 
                          string(v, "_", target),
                          sgn]
             )

    end
    
    return new_edges
end


function sifs_to_edgelists(pwy_sifs::Vector{<:Vector{<:Vector}})
    return map(sif_to_edgelist, pwy_sifs)
end


function sifs_to_edgelists(sif_files::Vector{<:AbstractString})
    sif_data = read_all_sif_files(sif_files)
    return sifs_to_edgelists(sif_data)
end



"""
    Given a set of proteins, construct an edgelist
    containing nodes/edges that represent the 
    central dogma for each gene.
    dna -> mrna -> protein -> activation
"""
function construct_dogma_edges(dogma_proteins; dogma_order=DOGMA_ORDER)

    n_prot = length(dogma_proteins)
    n_dogma = length(dogma_order)

    result = Vector{Any}(undef, n_prot*(n_dogma-1))

    k = 1
    for prot in dogma_proteins
        for (i, dog) in enumerate(dogma_order[1:end-1])
            result[k] = [string(prot, "_", dog), 
                         string(prot, "_", dogma_order[i+1]), 
                         1.0]
            k += 1
        end
    end

    return result
end


"""
    Given a vector of specially-formatted feature IDs 
    and a vector of connection weights, add
    edges that connect the features to appropriate parts
    of the central dogma.
"""
function construct_data_edges(feature_ids, feature_weights)

    n_feat = length(feature_ids)
    edges = Vector{Any}(undef, n_feat)
    for (i, feat) in enumerate(feature_ids)
        edges[i] = [feat, 
                    join(split(feat, "_")[1:2], "_"),
                    feature_weights[i]]
    end
    return edges
end


# Append relevant nodes from an edgelist with a given tag
function tag_nodes(edgelist; tag="_activation")

    result = deepcopy(edgelist)
    for edge in result
        edge[1] = string(edge[1], tag)
        edge[2] = string(edge[2], tag)
    end
    return result
end


"""
    Given a vector of pathway edgelists and a vector of data features,
    construct edgelists that (1) include "central dogma edges"
    and (2) edges that connect to the features.
"""
function extend_pathways(pwy_edgelists::Vector{Vector{T}} where T, 
                         feature_ids, feature_weights)

    feature_id_set = Set(feature_ids)

    feature_genes_dogmas = [split(fid,"_")[1:2] for fid in feature_ids]
    feature_genes = [p[1] for p in feature_genes_dogmas]
    feature_genes_set = Set(feature_genes)
    feature_dogmas = [p[2] for p in feature_genes_dogmas]

    n_pwy = length(pwy_edgelists)
    ext_edgelists = Vector{Vector{Any}}(undef, n_pwy)

    for (k, pwy) in enumerate(pwy_edgelists)

        pwy_genes = get_all_entities(pwy) 
        relevant_genes = intersect(pwy_genes, feature_genes_set)
        relevant_feature_idx = map(x -> in(x, relevant_genes), feature_genes)
        relevant_features = feature_ids[relevant_feature_idx]
        relevant_weights = feature_weights[relevant_feature_idx]

        # Construct central dogma edges for relevant proteins
        dogma_edgelist = construct_dogma_edges(relevant_genes)

        # Connect data features to the edgelist
        data_edges = construct_data_edges(relevant_features, relevant_weights)
        dogma_edgelist = vcat(dogma_edgelist, data_edges) 

        # Append an "_activation" tag to every pathway node
        all_edges = vcat(dogma_edgelist, pwy)

        # Recursively prune all leaves of the graph
        prune_leaves!(all_edges; except=feature_id_set)
        ext_edgelists[k] = all_edges
    end

    return ext_edgelists
end


function construct_pwy_feature_ids(feature_genes, feature_dogmas, feature_ids)
    return collect(string(g,"_",d,"_",i) for (g,d,i) in zip(feature_genes, feature_dogmas, feature_ids)) 
end


"""
    prep_pathway_graphs(pwy_sifs, feature_genes, feature_dogmas;
                        feature_weights=nothing)

    Given (1) a vector of pathway SIFs; 
    (2) a vector of feature genes; and
    (3) a vector of feature dogmas; 
    map the features into the pathways and construct 
    a vector of updated edgelists that include the features.

    Returns (1) a vector of new, augmented edgelists; and 
    (2) a new vector of feature IDs.

    `pwy_sifs`: a vector of paths to specially formatted SIF files;
                OR a vector of specially formatted edgelists.
    `feature_genes`: a vector of gene ID strings.
    `feature_dogmas`: a vector of strings belonging to
                      {"dna", "mrna", "protein"}.
    `feature_weights`: a vector of weights indicating
                       the sign/magnitude of the feature's
                       relationship to its dogma entity.
                       E.g., +1 -> a promoter relationship
                       and -1 -> a suppressor relationship. 
"""
function prep_pathway_graphs(pwy_sifs::Vector, feature_genes::Vector, feature_dogmas::Vector;
                             feature_ids::Union{Nothing,Vector}=nothing,
                             feature_weights::Union{Nothing,Vector}=nothing)

    valid_dogmas = Set(DOGMA_ORDER) 
    N = length(feature_genes)

    # Validate input
    @assert length(feature_dogmas) == N "`feature_genes` and `feature_dogmas` must have identical length"
    @assert all(map(x->in(x, valid_dogmas), feature_dogmas))
    if feature_weights == nothing
        feature_weights = ones(N)
    end
    if feature_ids == nothing
        feature_ids = collect(1:N)
    end 
   
    # Convert (genes, dogma-levels) to unique feature IDs 
    new_feature_ids = construct_pwy_feature_ids(feature_genes, feature_dogmas, feature_ids) 
    
    # Read the SIFs and translate to weighted graphs (edgelists)
    pwy_edgelists = sifs_to_edgelists(pwy_sifs) 

    # Use the central dogma to connect data features to the pathways
    pathway_graphs = extend_pathways(pwy_edgelists, new_feature_ids, feature_weights)    

    return pathway_graphs, new_feature_ids
end

###############################################################
# Construct featuresets
###############################################################

function sif_to_nodeset(pwy_sif::Vector)
    nodes = Set()
    for edge in pwy_sif
        push!(nodes, edge[1])
        push!(nodes, edge[3])
    end
    return nodes
end


function sifs_to_nodesets(pwy_sifs::Vector{<:Vector{<:Vector}})
    return map(sif_to_nodeset, pwy_sifs)
end

function sifs_to_nodesets(sif_files::Vector{<:AbstractString})
    sif_data = read_all_sif_files(sif_files)
    return map(sif_to_nodeset, sif_data)
end

"""
    prep_pathway_featuresets(pwy_sifs, feature_genes; feature_ids=nothing)

    Given (1) a vector of pathway SIFs and 
    (2) a vector of feature genes; and
    map the features into the pathways and construct 
    a vector of node sets that include the features.

    Returns (1) a vector of featuresets; and 
    (2) a vector of new feature IDs consistent with those featuresets.

    `pwy_sifs`: a vector of paths to specially formatted SIF files;
                OR a vector of specially formatted edgelists.
    `feature_genes`: a vector of gene ID strings.
"""
function prep_pathway_featuresets(pwy_sifs::Vector, feature_genes::Vector;
                                  feature_ids::Union{Nothing,Vector}=nothing)
    # Check whether feature ids were provided
    if feature_ids == nothing
        feature_ids = collect(1:length(feature_genes))
    end 

    # Create a map from genes to indices 
    gene_to_idxs = Dict()
    for (i,g) in enumerate(feature_genes)
        idx_set = get!(gene_to_idxs, g, Set())
        push!(idx_set, i)
    end

    # Create new feature ids 
    new_feature_ids = map(t->join(t, "_"), collect(zip(feature_genes, feature_ids)))
    n_pwy = length(pwy_sifs)

    # Convert the SIFs to node sets
    nodesets = sifs_to_nodesets(pwy_sifs)

    # For each pathway, find all of the features
    # associated with it
    feature_sets = Vector{Set}(undef, n_pwy)
    for (i,ns) in enumerate(nodesets)
        fs = Set()
        for node in ns
            node_idx = collect(get!(gene_to_idxs, node, []))
            union!(fs, new_feature_ids[node_idx])
        end 
        feature_sets[i] = fs
    end

    return feature_sets, new_feature_ids
end


