


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



#############################################################
# Model assembly
#############################################################


function assemble_model(pathway_sif_data,
                        pathway_names,
                        sample_ids, sample_conditions,
                        sample_batch_dict,
                        data_features,
                        lambda_X, lambda_Y;
                        model_features=nothing)

    if model_features == nothing
        model_features = sort_features(data_features)
    else
        @assert issubset(model_features, data_features)
        model_features = sort_features(model_features)
    end

    # Preprocess the pathways
    pathway_edgelists = prep_pathways(pathway_sif_data,
                                      model_features)

    # Sort the features
    data_features = zip(data_genes, data_assays)
    observed_nodes = intersect(data_features, pathway_nodes)
    srt_obs_nodes = sort_features(observed_nodes)
    srt_obs_genes = [get_gene(feat) for feat in srt_obs_nodes]
    srt_obs_assays = [get_assay(feat) for feat in srt_obs_nodes]

    # Track the feature permutation imposed by sorting
    _, perm_idx = keymatch(srt_obs_nodes, data_features)

    # Collect the batch ids
    unq_assays = unique(srt_obs_assays)
    sample_batch_ids = [sample_batch_dict[ua] for ua in unq_assays]

    # Get the loss model for each feature
    feature_losses = String[get_loss(feat) for feat in srt_obs_features]

    # Compute dimensions of the matrix factorization
    M = length(sample_ids)
    N = length(srt_obs_genes)
    K = length(pathway_sif_data)

    # Construct a regularizer for Y
    Y_reg = NetworkRegularizer(pathway_edgelists; observed=srt_obs_features,
                                                  weight=lambda_Y)

    # Construct regularizers for sigma and mu
    feature_group_edgelist = create_group_edgelist(srt_obs_features, srt_obs_assays)
    logsigma_reg = NetworkRegularizer([feature_group_edgelist]; observed=srt_obs_features,
                                                                weight=lambda_Y)
    mu_reg = NetworkRegularizer([feature_group_edgelist]; observed=srt_obs_features,
                                                          weight=lambda_Y)
    
    # Construct a regularizer for X
    sample_edgelist = create_group_edgelist(sample_ids, sample_conditions)
    X_reg = NetworkRegularizer(fill(sample_edgelist, K); observed=sample_ids,
                                                         weight=lambda_X)
    

    # Construct MatFacModel
    matfac = BatchMatFacModel(M, N, K, srt_obs_assays, 
                              sample_batch_ids, 
                              feature_losses;
                              X_reg=X_reg, Y_reg=Y_reg, 
                              logsigma_reg=logsigma_reg, 
                              mu_reg=mu_reg)

    pathway_weights = zeros(K)

    model = MultiomicModel(matfac, 
                           sample_ids, sample_conditions, 
                           perm_idx,
                           srt_obs_genes, srt_obs_assays,
                           pathway_names, pathway_weights)

    return model
end





