
using CSV
using DataFrames
using SparseArrays

export load_pathways, pathways_to_matrices, hierarchy_to_matrix 

##########################
# PATIENT PREPROCESSING
##########################


"""
    Given a nested dictionary of patients,
    generate an undirected graph representation
    of that tree; and then create a sparse matrix
    for it.
"""
function hierarchy_to_matrix(patient_hierarchy)

    edges = []
    all_nodes = []

    # Recursively build an undirected graph 
    # from the dictionary
    function rec_h2m(parent_name, children)
        push!(all_nodes, parent_name)
        if typeof(children) <: Vector
            for child_name in children
                push!(edges, [parent_name, child_name, 1.0])
                push!(edges, [child_name, parent_name, 1.0])
                push!(all_nodes, child_name)
            end
        elseif typeof(children) <: Dict
            for child_name in keys(children)
                push!(edges, [parent_name, child_name, 1.0])
                push!(edges, [child_name, parent_name, 1.0])
                rec_h2m(child_name, children[child_name])
            end
        end
    end
    rec_h2m("root", patient_hierarchy) 

    node_to_idx = Dict(v => idx for (idx,v) in enumerate(all_nodes))

    matrix = ugraph_to_matrix(edges, size(all_nodes,1), node_to_idx) 

    return (matrix, all_nodes)

end


##########################
# PATHWAY PREPROCESSING
##########################
DEFAULT_DATA_TYPES = ["cna","mutation",
                      "methylation","mrnaseq",
                      "rppa"]

DEFAULT_DATA_TYPE_MAP = Dict("cna" => ["dna", 1],
                             "mutation" => ["dna", -1],
                             "methylation" => ["mrna", -1],
                             "mrnaseq" => ["mrna", 1],
                             "rppa" => ["protein", 1]
                             )

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
    read a SIF file, returning it as a 
    vector of vectors
"""
function read_sif_file(sif_file::String)
    df = DataFrame!(CSV.File(sif_file))
    return [ collect(df[i,:]) for i=1:size(df,1) ]
end



"""
    read many SIF files, returning the result as a 
    Dictionary of vectors of vectors
"""
function read_all_sif_files(sif_files::Vector{String})
    return Dict(sp => read_sif_file(sp) for sp in sif_files)
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
        v = edge[3]
        attr = split(edge[2], "_")
        ent_types = attr[[1,3]]
        sgn = attr[2]
        target = attr[4]

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
        if sgn == "promote"
            sgn = 1
        elseif sgn == "suppress"
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


function add_data_nodes_sparse_latent(pathway,
                                      feature_map::Dict, 
                                      data_types, 
                                      data_node_map::Dict)

    # Get all of the proteins from this pathway
    proteins = Set{String}()
    for edge in pathway
        u = split(edge[1],"_")
        if u[end] == "protein"
            push!(proteins, u[1])
        end
    end
    # for each protein, add edges from data nodes to
    # to the correct parts of the graph
    for protein in proteins
        for dt in data_types
            for idx in feature_map[string(protein, "_", dt)]
                data_node = string(protein, "_", dt, "_", idx)
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


function convert_pwy_to_ugraph(pathway; strategy="symmetrize")

    if strategy == "symmetrize"
        reversed = [ [edge[2], edge[1], edge[3]] for edge in pathway ]
        ugraph = [pathway ; reversed] 
    else
        throw(DomainError(strategy, "not a valid option for `strategy`")) 
    end

    return ugraph
end


function ugraph_to_matrix(ugraph, N, node_to_idx, 
                          epsilon=nothing, 
                          standardize=true)

    if epsilon == nothing
        epsilon = 1.0 / sqrt(N)
    end

    I = []
    J = []
    V = []
    diag_entries = fill(epsilon, N)

    for edge in ugraph
        # Append off-diagonal entries
        push!(I, node_to_idx[edge[1]])
        push!(J, node_to_idx[edge[2]])
        push!(V, -1.0*edge[3])

        # Increment diagonal entries
        # (maintain diagonal dominance)
        diag_entries[node_to_idx[edge[1]]] += 1.0
    end

    # Now append diagonal entries
    for (i,v) in enumerate(diag_entries)
        push!(I, i)
        push!(J, i)
        push!(V, v)
    end 

    m = sparse(I, J, V, N, N)
   
    # optionally: rescale matrix s.t. we
    # have ones on the diagonal (i.e., the 
    # matrix is an inverted correlation matrix) 
    if standardize
        standardizer = sparse(1:N, 1:N, 1.0./sqrt.(diag_entries), N, N)
        m = standardizer * m * standardizer 
    end

    return m 
end


function convert_ugraphs_to_matrices(ugraphs)

    # Get all of the nodes from all of the graphs
    all_nodes = Set{String}()
    for (name,u) in ugraphs
        for edge in u
            push!(all_nodes, edge[1])
        end
    end
    # in lexicographic order
    all_nodes = sort(collect(all_nodes))
    N = size(all_nodes,1)

    # create a mapping from node names to indices
    node_to_idx = Dict(v => i for (i, v) in enumerate(all_nodes))

    # now translate the ugraphs to sparse matrices
    matrices = Dict(name => ugraph_to_matrix(u, N, node_to_idx) for (name,u) in ugraphs)

    return matrices, all_nodes
end


function ugraphs_to_regularizers(ugraphs)

    matrices, all_nodes = convert_ugraphs_to_matrices(ugraphs)

    pwy_names = sort(collect(keys(matrices)))

    regs = RowReg[ RowReg(zeros(k), 
                   Vector{Tuple{Int64,Int64,Float64}}(), 1.0) 
                   for node in all_nodes]

    for (k, pwy_name) in enumerate(pwy_names)
        mat = matrices[pwy_name]
        I, J, V = findnz(mat)
        for (idx, i) in enumerate(I)
            push!(regs[i].neighbors, (J[idx], k, V[i])) 
        end
    end

    return regs, pwy_names

end



"""
    Given a dictionary of pathways and a populated featuremap,
    return a corresponding dictionary of sparse matrices
    and an array that maps indices to pathway nodes
"""
function pathways_to_matrices(pathway_dict, featuremap;
                              data_types=DEFAULT_DATA_TYPES, 
                              data_type_map=DEFAULT_DATA_TYPE_MAP,
                              pwy_data_augmentation="sparse_latent",
                              pwy_to_ugraph="symmetrize")

    # Augment the pathways with additional nodes
    # to represent the data    
    for (name,pwy) in pathway_dict
        pathway_dict[name] = add_data_nodes_to_pathway(pwy, featuremap, data_types, data_type_map; 
                                                       strategy=pwy_data_augmentation)
    end

    # Pathways are currently interpreted as
    # directed graphs. Convert to undirected graphs
    for (name, pwy) in pathway_dict
        pathway_dict[name] = convert_pwy_to_ugraph(pwy; strategy=pwy_to_ugraph)
    end

    # Translate the undirected graphs into sparse matrices
    matrices, idx_decoder = convert_ugraphs_to_matrices(pathway_dict)
 
    return (matrices, idx_decoder)
end




function get_all_proteins(pathways_dict)

    proteins = Set{String}()
    for pair in pathways_dict
        for edge in pair.second
            tok = split(edge[1],"_")
            if tok[2] == "protein"
                push!(proteins, tok[1])
            end
        end
    end

    return proteins
end



function load_pathways(sif_filenames, data_kinds)

    pathways = read_all_sif_files(sif_filenames)
   
    extended_pwys = Dict() 
    for (name, pwy) in pathways
        extended_pwys[name] = extend_pathway(pwy)
    end
    
    all_proteins = get_all_proteins(extended_pwys)

    empty_feature_map = initialize_featuremap(all_proteins, data_kinds)

    return (extended_pwys, empty_feature_map)
end 


