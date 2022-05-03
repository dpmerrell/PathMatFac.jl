


######################################################
# Pathway SIF file input
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

######################################################
# Construct the edgelists
######################################################

"""
    Convert a SIF pathway description to a
    list of edges in a more useful format.
"""
function sif_to_edgelist(pwy_sif::Vector)

    new_edges = Vector{Any}[] 

    # Modify the edges to/from proteins
    for edge in pwy_sif 
        u = edge[1]
        
        ent_types = (PWY_SIF_CODE[edge[2][[1]]], 
                     PWY_SIF_CODE[edge[2][[2]]]
                    )
        target = PWY_SIF_CODE[edge[2][[3]]]
        sgn = PWY_SIF_CODE[edge[2][[4]]]
        
        v = edge[3]

        # If u is a protein, replace
        # it with an "activation" node
        if ent_types[1] == "protein"
            new_u = string(u, "_activation")
        else
            new_u = string(u, "_", ent_types[1])
        end  

        # If v is a protein, check whether
        # its transcription or activation is
        # targeted by this edge
        if ent_types[2] == "protein"
            if target == "transcription"
                new_v = string(v, "_mrna")
            else
                new_v = string(v, "_activation")
            end
        else
            new_v = string(v, "_", ent_types[2])
        end

        push!(new_edges, [(new_u,""), (new_v,""), sgn])
    end
    
    return new_edges
end


"""
    Given a set of genes, construct an edgelist
    containing nodes/edges that represent the 
    central dogma for each gene.
    dna -> mrna -> protein -> activation
"""
function construct_dogma_edges(unq_dogma_genes)

    edgelist = Vector{Any}[]
    for genes in unq_dogma_genes
        gene_names = split(genes, " ")
        for g in gene_names
            push!(edgelist, [(string(g, "_dna"), ""), (string(g, "_mrna"), ""), 1])
            push!(edgelist, [(string(g, "_mrna"), ""), (string(g, "_protein"), ""), 1])
            push!(edgelist, [(string(g, "_protein"), ""), (string(g, "_activation"), ""), 1])
        end
    end

    return edgelist
end


"""
    Given an edgelist and a vector of features, add
    edges that connect the features to appropriate parts
    of the existing network.
"""
function construct_data_edges(dogma_features; assay_map=DEFAULT_ASSAY_MAP)

    edges = Vector{Any}[]
    for (genes, assay) in dogma_features        
        suff, weight = assay_map[assay]
        gene_names = split(genes, " ")
        for g in gene_names
            push!(edges, [(genes, assay), (string(g,"_",suff), ""), weight])
        end
    end
    return edges
end


"""
    Given a vector of pathway SIFs and a vector of data features,
    construct edgelists that represent (1) the pathways and 
    (2) those features' relationships to the central dogma.
"""
function prep_pathways(pwy_sif_data::Vector{Vector{T}} where T, features;
                       dogma_features=nothing,
                       assay_map=DEFAULT_ASSAY_MAP)

    # Determine which features will be included
    # in the "central dogma" network
    if dogma_features == nothing
        dogma_features = copy(features)
    else
        @assert issubset(dogma_features, features)
    end

    # Construct the "central dogma" edgelist for these features
    unq_dogma_genes = unique(map(get_gene, features))
    dogma_edgelist = construct_dogma_edges(unq_dogma_genes)

    # Add data edges to the edgelist
    data_edges = construct_data_edges(dogma_features; assay_map=assay_map)
    dogma_edgelist = vcat(dogma_edgelist, data_edges) 

    # Construct the pathway-specific graphs
    pwy_edgelists = Vector{Any}[]
    for pwy in pwy_sif_data
        edgelist = deepcopy(dogma_edgelist)
        edgelist = vcat(edgelist, sif_to_edgelist(pwy)) 

        prune_leaves!(edgelist; except=features)
        push!(pwy_edgelists, edgelist)
    end

    return pwy_edgelists
end


function prep_pathways(pathway_files::Vector{String}, features;
                       kwargs...)

    pwy_sif_data = read_all_sif_files(pathway_files) 
    return prep_pathways(pwy_sif_data, features; kwargs...)
end


