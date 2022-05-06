


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


function sifs_to_edgelists(pwy_sifs::Vector{<:Vector{<:Vector}})
    return map(sif_to_edgelist, pwy_sifs)
end


function sifs_to_edgelists(sif_files::Vector{<:AbstractString})
    sif_data = read_all_sif_files(sif_files)
    return sifs_to_edgelists(sif_data)
end

###TODO: REORGANIZE CODE TO AVOID ~30MINUTE PREPROCESSING TIMES!!! :(
###"""
###    Given a set of genes, construct an edgelist
###    containing nodes/edges that represent the 
###    central dogma for each gene.
###    dna -> mrna -> protein -> activation
###"""
###function construct_dogma_edges(unq_dogma_genes)
###
###    edgelist = Vector{Any}[]
###    for genes in unq_dogma_genes
###        gene_names = split(genes, " ")
###        for g in gene_names
###            push!(edgelist, [(string(g, "_dna"), ""), (string(g, "_mrna"), ""), 1])
###            push!(edgelist, [(string(g, "_mrna"), ""), (string(g, "_protein"), ""), 1])
###            push!(edgelist, [(string(g, "_protein"), ""), (string(g, "_activation"), ""), 1])
###        end
###    end
###
###    return edgelist
###end


"""
    Given a set of features, construct an edgelist
    containing nodes/edges that represent the 
    central dogma for each gene.
    dna -> mrna -> protein -> activation
"""
function construct_dogma_edges(dogma_features; assay_map=DEFAULT_ASSAY_MAP)

    dogma_to_idx = value_to_idx(DOGMA_ORDER)

    # Find the lowest and highest 
    dogmax = Dict()
    dogmin = Dict()
    for (gene, assay) in features
        level = assay_map[assay][1]
        dogmin[gene] = min(dogmin[gene], level)
        dogmax[gene] = max(dogmax[gene], level)
    end

    edgelist = Vector{Any}[]
    for (gene, mn) in dogmin
        mx = dogmax[gene]
        for i=mn:(mx-1)
            push!(edgelist, [string(gene, "_", DOGMA_ORDER[i]),
                             string(gene, "_", DOGMA_ORDER[i+1]), 1])
        end
    end

    return edgelist, dogmax
end

"""
    Given an edgelist and a vector of features, add
    edges that connect the features to appropriate parts
    of the existing network.
"""
function construct_data_edges(dogma_features; assay_map=DEFAULT_ASSAY_MAP)

    edges = Vector{Any}[]
    for (gene, assay) in dogma_features        
        suff, weight = assay_map[assay]
        push!(edges, [(gene, assay), (string(g,"_",suff), ""), weight])
    end
    return edges
end


function connect_pwy_to_dogma!(dogma_edges, pwy, dogmax)

    dogma_to_idx = value_to_idx(DOGMA_ORDER)

    pwy_nodes = get_all_nodes(pwy)
    pwy_node_splits = map(x->split(x[1],"_"), pwy_nodes)
    pwy_proteins = [node[1] for node in pwy_node_splits if node[2]=="activation"]

    activation_idx = dogma_to_idx["activation"]
    termination = activation_idx - 1

    for prot in pwy_proteins
        if prot in keys(dogmax)
            for i=(dogma_to_idx[dogmax[prot]]):termination
                push!(dogma_edges, [(string(prot,"_",DOGMA_ORDER[i]), ""), 
                                    (string(prot,"_",DOGMA_ORDER[i+1]), ""), 1])
            end
        end
    end

    dogma_edges = vcat(dogma_edges, pwy)
end

"""
    Given a vector of pathway edgelists and a vector of data features,
    construct edgelists that (1) include "central dogma edges"
    and (2) edges that connect to the features.
"""
function extend_pathways(pwy_edgelists::Vector{Vector{T}} where T, features;
                         dogma_features=nothing,
                         assay_map=DEFAULT_ASSAY_MAP)

    # Determine which features will be included
    # in the "central dogma" network
    if dogma_features == nothing
        dogma_features = copy(features)
    else
        @assert issubset(dogma_features, features)
    end

    ###TODO: REORGANIZE CODE TO AVOID ~30MINUTE PREPROCESSING TIMES!!! :(
    ##### Construct the "central dogma" edgelist for these features
    ####unq_dogma_genes = unique(map(get_gene, dogma_features))
    ####dogma_edgelist = construct_dogma_edges(unq_dogma_genes)

    # Construct "central dogma" edges for these features
    dogma_edgelist, 
    dogma_max_levels = construct_dogma_edges(dogma_features; assay_map=assay_map)
    # `dogma_max_levels` is a dictionary:
    # gene => highest level of central dogma with data for that gene

    # Add data edges to the edgelist
    data_edges = construct_data_edges(dogma_features; assay_map=assay_map)
    dogma_edgelist = vcat(dogma_edgelist, data_edges) 

    # Construct the pathway-specific graphs
    ext_edgelists = Vector{Vector{Any}}[]
    for pwy in pwy_edgelists
        full_edgelist = deepcopy(dogma_edgelist)
        connect_pwy_to_dogma!(full_edgelist, pwy, dogma_max_levels)
        
        ###TODO: REORGANIZE CODE TO AVOID ~30MINUTE PREPROCESSING TIMES!!! :(
        ###full_edgelist = vcat(full_edgelist, pwy) 
        ####prune_leaves!(full_edgelist; except=features)
        push!(ext_edgelists, full_edgelist)
    end

    return ext_edgelists
end




