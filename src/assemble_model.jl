


#############################################################
# Identify L1-regularized features for each factor
#############################################################

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
                        l1_fraction=0.5,
                        lambda_layer=0.1,
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

    # Construct a regularizer for Y.
    # We apply L1 regularization to non-pathway entries.
    l1_features = compute_nonpwy_features(model_features, pwy_edgelists)

    println("\tConstructing regularizers...")
    Y_reg = NetworkL1Regularizer(model_features, ext_edgelists, l1_features;
                                 epsilon=1.0, 
                                 net_weight=2.0*(1 - l1_fraction),
                                 l1_weight=2.0*l1_fraction)

    # Construct the column layers
    col_layers = PMLayers(model_assays, sample_batch_ids) 

    # Construct a regularizer for the column layers
    logsigma_reg = ClusterRegularizer(model_assays; weight=1.0)
    mu_reg = ClusterRegularizer(model_assays; weight=1.0)
    logdelta_reg = BatchArrayReg(col_layers.bscale.logdelta; 
                                 weight=1.0)
    theta_reg = BatchArrayReg(col_layers.bshift.theta;
                              weight=1.0)

    layer_reg = PMLayerReg(logsigma_reg, mu_reg, logdelta_reg, theta_reg) 

    # Construct a regularizer for X
    X_reg = ClusterRegularizer(sample_conditions)

    # Construct MatFacModel
    matfac = MatFacModel(M, N, K, feature_losses;
                         col_transform=col_layers,
                         X_reg=X_reg, Y_reg=Y_reg, 
                         col_transform_reg=layer_reg,
                         lambda_X=lambda_X, lambda_Y=lambda_Y,
                         lambda_col=lambda_layer)

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





