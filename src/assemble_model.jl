


#############################################################
# Patient groups
#############################################################

function create_group_edgelist(id_vec, group_vec; rooted=false)
    
    m = length(id_vec)
    @assert m == length(group_vec)

    edgelist = Vector{Any}[]

    # Tie samples to their groups
    for i=1:m
        push!(edgelist, [group_vec[i], id_vec[i], 1])
    end

    # If we're rooted, then tie the groups to a root node
    if rooted
        for gp in unique(group_vec)
            push!(edgelist, ["ROOT", gp, 1])
        end
    end

    return edgelist
end

function compute_nonpwy_features(model_features, pathway_edgelists)

    result = []
    for el in pathway_edgelists
        pwy_nodes = get_all_nodes(el)
        pwy_proteins = Set([split(node[1],"_")[1] for node in pwy_nodes])
        
        l1_feat = [mf for mf in model_features if !in(mf[1], pwy_proteins)]
        push!(result, l1_feat)
    end

    return result
end


#############################################################
# Model assembly
#############################################################

function assemble_model(pathway_sif_data,
                        pathway_names,
                        sample_ids, sample_conditions,
                        sample_batch_dict,
                        data_features,
                        lambda_X, lambda_Y;
                        model_features=nothing,
                        dogma_features=nothing)

    # By default, the model uses all data features. 
    if model_features == nothing
        model_features = sort_features(data_features)
    else
        @assert issubset(model_features, data_features)
        model_features = sort_features(model_features)
    end
    
    # Bookkeeping to track the subsetting/sorting
    _, used_feature_idx = keymatch(model_features, data_features)

    # Extract genes and assays from the features
    model_genes = [get_gene(feat) for feat in model_features]
    model_assays = [get_assay(feat) for feat in model_features]

    # Collect the batch ids
    unq_assays = unique(model_assays)
    sample_batch_ids = [sample_batch_dict[ua] for ua in unq_assays]

    # Get the loss model for each feature
    feature_losses = String[get_loss(feat) for feat in model_features]

    # Compute dimensions of the matrix factorization
    M = length(sample_ids)
    N = length(model_genes)
    K = length(pathway_sif_data)
    
    # Preprocess the pathways
    println("\tPreprocessing pathways...")
    pwy_edgelists = sifs_to_edgelists(pathway_sif_data)
    ext_edgelists = extend_pathways(pwy_edgelists,
                                    model_features;
                                    dogma_features=dogma_features)

    # Construct a regularizer for Y
    nonpwy_features = compute_nonpwy_features(model_features, pwy_edgelists)

    println("\tConstructing regularizers...")

    Y_reg = NetworkL1Regularizer(model_features, ext_edgelists; 
                                 net_weight=lambda_Y, l1_weight=lambda_Y,
                                 l1_features=nonpwy_features)

    # Construct regularizers for sigma and mu
    feature_group_edgelist = create_group_edgelist(model_features, model_assays)
    logsigma_reg = NetworkRegularizer([feature_group_edgelist]; observed=model_features,
                                                                weight=lambda_Y)
    mu_reg = NetworkRegularizer([feature_group_edgelist]; observed=model_features,
                                                          weight=lambda_Y)
    
    # Construct a regularizer for X
    sample_edgelist = create_group_edgelist(sample_ids, sample_conditions)
    X_reg = NetworkRegularizer(fill(sample_edgelist, K); observed=sample_ids,
                                                         weight=lambda_X)
    

    # Construct MatFacModel
    matfac = BatchMatFacModel(M, N, K, model_assays, 
                              sample_batch_ids, 
                              feature_losses;
                              X_reg=X_reg, Y_reg=Y_reg, 
                              logsigma_reg=logsigma_reg, 
                              mu_reg=mu_reg)

    pathway_weights = zeros(K)

    data_genes = [get_gene(feat) for feat in data_features]
    data_assays = [get_assay(feat) for feat in data_features]

    model = MultiomicModel(matfac, 
                           sample_ids, sample_conditions, 
                           data_genes, data_assays,
                           used_feature_idx,
                           pathway_names, pathway_weights)

    return model
end





