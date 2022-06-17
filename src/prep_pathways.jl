


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

    # Extract sign and target attributes
    # from the interaction
    for edge in pwy_sif 
        
        u = edge[1]
        
        target = PWY_SIF_CODE[edge[2][[1]]]
        sgn = PWY_SIF_CODE[edge[2][[2]]]

        v = edge[3]

        push!(new_edges, [(u, PWY_SIF_CODE["a"]), 
                          (v, target), sgn])

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
    Given a set of features, construct an edgelist
    containing nodes/edges that represent the 
    central dogma for each gene.
    dna -> mrna -> protein -> activation
"""
function construct_dogma_edges(dogma_features; assay_map=DEFAULT_ASSAY_MAP,
                                               dogma_to_idx=DOGMA_TO_IDX)

    # Find the lowest and highest 
    dogmax = Dict()
    dogmin = Dict()
    for (gene, assay) in dogma_features
        level = dogma_to_idx[assay_map[assay][1]]
        if haskey(dogmax, gene)
            dogmin[gene] = min(dogmin[gene], level)
            dogmax[gene] = max(dogmax[gene], level)
        else
            dogmin[gene] = level
            dogmax[gene] = level
        end
    end

    # Construct edges between levels of
    # the central dogma touched by the data
    edgelist = Vector{Any}[]
    for (gene, mn) in dogmin
        mx = dogmax[gene]
        for i=mn:(mx-1)
            push!(edgelist, [(gene, DOGMA_ORDER[i]),
                             (gene, DOGMA_ORDER[i+1]), 1])
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
        push!(edges, [(gene, assay), (gene, suff), weight])
    end
    return edges
end


function connect_pwy_to_dogma(dogma_edges, pwy, dogmax, full_geneset;
                              dogma_to_idx=DOGMA_TO_IDX)

    pwy_nodes = get_all_nodes(pwy)
    pwy_proteins = [node[1] for node in pwy_nodes if node[1] in full_geneset]

    activation_idx = dogma_to_idx["activation"]
    termination = activation_idx - 1

    # Connect existing dogma edges to "activation"
    # by adding missing edges
    for prot in pwy_proteins
        if haskey(dogmax, prot)
            for i=(dogmax[prot]):termination
                push!(dogma_edges, [(prot, DOGMA_ORDER[i]), 
                                    (prot, DOGMA_ORDER[i+1]), 1])
            end
        end
    end

    return vcat(dogma_edges, pwy)
end


"""
    Given a vector of pathway edgelists and a vector of data features,
    construct edgelists that (1) include "central dogma edges"
    and (2) edges that connect to the features.
"""
function extend_pathways(pwy_edgelists::Vector{Vector{T}} where T, 
                         features;
                         dogma_features=nothing,
                         assay_map=DEFAULT_ASSAY_MAP)

    # Determine which features will be included
    # in the "central dogma" network
    if dogma_features == nothing
        dogma_features = copy(features)
    else
        @assert issubset(dogma_features, features)
    end

    # Construct "central dogma" edges for these features
    dogma_edgelist, 
    dogma_max_levels = construct_dogma_edges(dogma_features; assay_map=assay_map)
    # `dogma_max_levels` is a dictionary:
    # gene => highest level of central dogma with data for that gene

    # Add data edges to the edgelist
    data_edges = construct_data_edges(dogma_features; assay_map=assay_map)
    dogma_edgelist = vcat(dogma_edgelist, data_edges) 

    full_geneset = Set([gene for (gene,_) in dogma_features])

    # Construct the pathway-specific graphs
    ext_edgelists = Vector{Vector{Any}}[]
    for pwy in pwy_edgelists
        full_edgelist = copy(dogma_edgelist)
        full_edgelist = connect_pwy_to_dogma(full_edgelist, pwy, dogma_max_levels, full_geneset)
        
        push!(ext_edgelists, full_edgelist)
    end

    return ext_edgelists
end


