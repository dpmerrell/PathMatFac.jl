
using CSV
using DataFrames
using SparseArrays

export load_pathway_sifs, load_pathways, pathways_to_ugraphs, 
       hierarchy_to_regularizers,
       get_instance_hierarchy, feature_to_loss, 
       DEFAULT_OMICS, DEFAULT_OMIC_MAP



DEFAULT_OMICS = ["cna",
                 "mutation",
                 "methylation",
                 "mrnaseq",
                 "rppa"]

DEFAULT_OMIC_DTYPES = Dict("cna" => Float64,
                           "mutation" => Bool,
                           "methylation" => Float64,
                           "mrnaseq" => Float64, 
                           "rppa" => Float64
                         )

DEFAULT_OMIC_MAP = Dict("cna" => ["dna", 1],
                             "mutation" => ["dna", -1],
                             "methylation" => ["mrna", -1],
                             "mrnaseq" => ["mrna", 1],
                             "rppa" => ["protein", 1]
                             )

function value_to_idx(values)
    return Dict(v => idx for (idx, v) in enumerate(values))
end

function keymatch(l_keys, r_keys)

    rkey_to_idx = value_to_idx(r_keys) 

    l_idx = []
    r_idx = []

    for (i, lk) in enumerate(l_keys)
        if lk in keys(rkey_to_idx)
            push!(l_idx, i)
            push!(r_idx, rkey_to_idx[lk])
        end
    end

    return l_idx, r_idx
end


###################################
# Model assembly
###################################

function assemble_matrix(data, features, extended_features,
                               instances, extended_instances)

    # Map the rows of the dataset
    # to the rows of the output matrix
    matrix_cols, data_cols = keymatch(extended_features, features)
    #matrix_cols = Vector{Int}() 
    #data_cols = Vector{Int}()

    #feature_set = Set(features)
    #feat_to_data_idx = Dict( feat => idx for (idx, feat) in enumerate(features) )

    #for (i, ext_feat) in enumerate(extended_features)
    #    if ext_feat in feature_set 
    #        idx = feat_to_data_idx[ext_feat]
    #        push!(matrix_cols, i)
    #        push!(data_cols, idx)
    #    end
    #end

    # Ignore the "artificial" instances: 
    # i.e., the fictitious hidden nodes
    # in the instance tree.
    matrix_rows, data_rows = keymatch(extended_instances, instances)
    #real_instances = intersect(Set(extended_instances), Set(instances))
    #real_instance_vec = String[inst for inst in extended_instances if inst in real_instances]

    ## Map the instances to the rows 
    ## of the output matrix
    #instance_to_matrow = value_to_idx(real_instance_vec) # Dict(inst => idx for (idx, inst) in enumerate(real_instance_vec))
    #matrix_rows = Int64[instance_to_matrow[inst] for inst in real_instance_vec]
    #matrix_rows = keymatch()

    ## Map the instances to the columns
    ## of the HDF file
    #instance_to_datarow = value_to_idx(instances) # Dict(inst => idx for (idx, inst) in enumerate(instances))
    #data_rows = Int64[instance_to_datarow[pat] for pat in real_instance_vec]
    
    # Initialize the matrix!
    result = fill(NaN, size(extended_instances,1),
                       size(extended_features,1))
    result = convert(Matrix{Number}, result)


    # Finally: load the data!    
    result[matrix_rows, matrix_cols] = data[data_rows, data_cols]

    set_matrix_dtypes!(result, matrix_cols, extended_features)

    return result
end


function feature_to_loss(feature_name)
    omic_type = split(feature_name, "_")[end]
    if DEFAULT_OMIC_DTYPES[omic_type] <: Bool
        return LogisticLoss()
    #elseif DEFAULT_OMIC_DTYPES[omic_type] <: Float64
    #    return QuadLoss()
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


##########################
# INSTANCE PREPROCESSING
##########################

function get_instance_hierarchy(instance_ids::Vector{T}, 
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


function hierarchy_to_regularizers(instance_hierarchy, k; fixed_instances=[], offset=false)

    matrix, ext_instances = hierarchy_to_matrix(instance_hierarchy)

    matrices = fill(matrix, k)

    regularizers = matrices_to_regularizers(matrices, ext_instances; 
                                            fixed_nodes=fixed_instances,
                                            offset=offset)
    
    return regularizers, ext_instances
end


##########################
# PATHWAY PREPROCESSING
##########################

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
    vector of vectors of vectors
"""
function read_all_sif_files(sif_files::Vector{String})
    return [read_sif_file(sp) for sp in sif_files]
end


PWY_SIF_CODE = Dict("a" => "activation",
                    "b" => "abstract",
                    "c" => "compound",
                    "h" => "chemical",
                    "p" => "protein",
                    "f" => "family"
               )

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
        #attr = edge[2] #split(edge[2], "_")
        #ent_types = attr[[1,2]]
        #target = attr[3]
        #sgn = attr[[end]]

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
                data_node = string(protein, "_", dt) #, "_", idx)
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


function extend_losses(losses, features, ext_features)

    ext_losses = fill(QuadLoss(), size(ext_features,1))
    ext_losses = convert(Vector{Loss}, ext_losses)

    feat_to_loss = Dict(feat => loss for (feat, loss) in zip(features, losses))

    for (i, feat) in enumerate(ext_features)
        if feat in keys(feat_to_loss)
            ext_losses[i] = feat_to_loss[feat]
        end
    end

    return ext_losses
end


function ugraph_to_matrix(graph::ElUgraph, node_to_idx, 
                          epsilon=nothing, 
                          standardize=true)
    N = length(node_to_idx)

    if epsilon == nothing
        epsilon = 1.0 / sqrt(N)
    end

    I = Int64[]
    J = Int64[]
    V = Float64[]
    diag_entries = fill(epsilon, N)

    for edge in graph.edges
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


function ugraphs_to_matrices(ugraphs::Vector{ElUgraph})

    # Get all of the nodes from all of the graphs
    all_nodes = Set{String}()
    for graph in ugraphs
        for edge in graph.edges
            push!(all_nodes, edge[1])
        end
    end

    # in lexicographic order
    all_nodes = sort(collect(all_nodes))
    N = size(all_nodes,1)

    # create a mapping from node names to indices
    node_to_idx = value_to_idx(all_nodes) # Dict(v => i for (i, v) in enumerate(all_nodes))

    # now translate the ugraphs to sparse matrices
    matrices = [ugraph_to_matrix(u, node_to_idx) for u in ugraphs]

    return matrices, all_nodes
end


function matrices_to_regularizers(matrices, all_nodes; fixed_nodes=[], offset=false)

    k = length(matrices)
    latent_dim = k

    fixed_node_set = Set(fixed_nodes)

    regs = Regularizer[]
    for (idx, node) in enumerate(all_nodes)
        if node in fixed_node_set
            push!(regs, FixedColReg(zeros(latent_dim), idx))
        else
            push!(regs, RowReg(zeros(latent_dim), Vector{Tuple{Int64,Int64,Float64}}(), 1.0)) 
        end
    end

    for (mat_idx, mat) in enumerate(matrices)
        I, J, V = findnz(mat)
        for (idx, i) in enumerate(I)
            j = J[idx]
            if !in(all_nodes[i], fixed_node_set) & (i != j)
                push!(regs[i].neighbors, (j, mat_idx, V[i]))
            end 
        end
    end

    return regs

end


"""
    Given a vector of pathways and a populated featuremap,
    return a corresponding dictionary of sparse matrices
    and an array that maps indices to pathway nodes
"""
function pathways_to_ugraphs(pathways, featuremap;
                             data_types=DEFAULT_OMICS, 
                             data_type_map=DEFAULT_OMIC_MAP,
                             pwy_data_augmentation="sparse_latent",
                             pwy_to_ugraph="symmetrize")

    # Augment the pathways with additional nodes
    # to represent the data    
    ext_pathways = [add_data_nodes_to_pathway(pwy, featuremap, data_types, data_type_map;
                                              strategy=pwy_data_augmentation)
                                              for pwy in pathways]

    # Pathways are currently interpreted as
    # directed graphs. Convert to undirected graphs
    ugraphs = ElUgraph[construct_elugraph(pwy) for pwy in ext_pathways]

    return ugraphs 
end


function ugraphs_to_regularizers(ugraphs::Vector{ElUgraph}; offset=false)

    matrices, ext_features = ugraphs_to_matrices(ugraphs)

    regularizers = matrices_to_regularizers(matrices, ext_features; offset=offset)

    return regularizers, ext_features
end



function get_all_proteins(pathways)

    proteins = Set{String}()
    for edge_list in pathways
        for edge in edge_list
            tok = split(edge[1],"_")
            if tok[2] == "protein"
                push!(proteins, tok[1])
            end
        end
    end

    return proteins
end


function load_pathways(pwy_vec, data_kinds)

    extended_pwys = [extend_pathway(pwy) for pwy in pwy_vec]
    all_proteins = get_all_proteins(extended_pwys)
    empty_feature_map = initialize_featuremap(all_proteins, data_kinds)

    return (extended_pwys, empty_feature_map)
end


function load_pathway_sifs(sif_filenames, data_kinds)

    pathways = read_all_sif_files(sif_filenames)
    extended_pwys, empty_feature_map = load_pathways(pathways, data_kinds)
 
    return (extended_pwys, empty_feature_map)
end 


