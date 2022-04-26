


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
                        feature_genes, feature_assays,
                        lambda_X, lambda_Y)

    # Sort the features
    features = zip(feature_genes, feature_assays)
    srt_features = sort_features(features)
    srt_genes = [get_gene(feat) for feat in srt_features]
    srt_assays = [get_assay(feat) for feat in srt_features]

    # Track the feature permutation imposed by sorting
    _, perm_idx = keymatch(srt_features, features)

    # Collect the batch ids
    unq_assays = unique(srt_assays)
    sample_batch_ids = [sample_batch_dict[ua] for ua in unq_assays]

    # Get the loss model for each feature
    feature_losses = String[get_loss(feat) for feat in srt_features]

    # Compute dimensions of the matrix factorization
    M = length(sample_ids)
    N = length(feature_genes)
    K = length(pathway_sif_data)

    # Construct a regularizer for X
    sample_edgelist = create_group_edgelist(sample_ids, sample_conditions)
    X_reg = NetworkRegularizer(fill(sample_edgelist, K); observed=sample_ids,
                                                         weight=lambda_X)

    # Construct the pathway regularizer for Y
    pathway_edgelists, pathway_nodes = prep_pathways(pathway_sif_data,
                                                     feature_genes,
                                                     feature_assays)
    Y_reg = NetworkRegularizer(pathway_edgelists; observed=srt_features,
                                                  weight=lambda_Y)

    # Construct regularizers for sigma and mu
    feature_group_edgelist = create_group_edgelist(srt_features, srt_assays)
    logsigma_reg = NetworkRegularizer([feature_group_edgelist]; observed=srt_features,
                                                                weight=lambda_Y)
    mu_reg = NetworkRegularizer([feature_group_edgelist]; observed=srt_features,
                                                          weight=lambda_Y)

    # Construct MatFacModel
    matfac = BatchMatFacModel(M, N, K, srt_assays, 
                              sample_batch_ids, 
                              feature_losses;
                              X_reg=X_reg, Y_reg=Y_reg, 
                              logsigma_reg=logsigma_reg, 
                              mu_reg=mu_reg)

    pathway_weights = zeros(K)

    model = MultiomicModel(matfac, 
                           sample_ids, sample_conditions, 
                           perm_idx,
                           srt_genes, srt_assays,
                           pathway_names, pathway_weights)

    return model
end





