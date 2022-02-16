


#############################################################
# Patient groups
#############################################################

function augment_samples(sample_ids, group_ids; rooted=false)
    result = vcat(sample_ids, unique(group_ids))
    if rooted
        push!(result, "ROOT")
    end
    return result
end


function create_sample_edgelist(sample_id_vec, group_vec; rooted=false)
    
    m = length(sample_id_vec)
    @assert m == length(group_vec)

    edgelist = Vector{Any}[]

    # Tie samples to their groups
    for i=1:m
        push!(edgelist, [group_vec[i], sample_id_vec[i], 1])
    end

    # If we're rooted, then tie the groups to a root node
    if rooted
        for gp in unique(group_vec)
            push!(edgelist, ["ROOT", gp, 1])
        end
    end

    return edgelist
end


function assemble_sample_reg_mat(sample_ids, sample_conditions)
    
    # build the sample edge list from the "sample groups" vector
    augmented_samples = augment_samples(sample_ids, sample_conditions) 
    sample_edgelist = create_sample_edgelist(sample_ids, sample_conditions)
    aug_sample_to_idx = value_to_idx(augmented_samples)

    # translate the sample edge list to a sparse matrix
    sample_reg_mat = edgelist_to_spmat(sample_edgelist, aug_sample_to_idx)

    return sample_reg_mat, augmented_samples, aug_sample_to_idx
end


function update_sample_batch_dict(sample_batch_dict::Dict{T,Vector{U}}, 
                                  sample_ids, internal_samples,
                                  internal_sample_to_idx) where T where U
  
    old_samples_set = Set(sample_ids)
    new_M = length(internal_samples)

    new_dict = Dict{T,Vector{U}}()


    for k in keys(sample_batch_dict)
        
        old_batches_lookup = Dict(zip(sample_ids, sample_batch_dict[k]))

        new_batches = Vector{T}(undef, new_M)

        for (idx, i_samp) in enumerate(internal_samples)
            if i_samp in old_samples_set
                new_batches[idx] = old_batches_lookup[i_samp] 
            else
                new_batches[idx] = ""
            end
        end

        new_dict[k] = new_batches
    end

    new_dict[""] = repeat([""], new_M)

    return new_dict 
end


##############################################################
## Model assembly
##############################################################




function assemble_model(pathway_sif_data,  
                        sample_ids, sample_conditions,
                        sample_batch_dict,
                        feature_genes, feature_assays,
                        lambda_X, lambda_Y)

    K = length(pathway_sif_data)
    lambda_X = BMF.BMFFloat(lambda_X)
    lambda_Y = BMF.BMFFloat(lambda_Y)

    # Construct the sample regularizer matrix (for X)
    sample_reg_mat, 
    internal_samples, 
    internal_sample_to_idx = assemble_sample_reg_mat(sample_ids, 
                                                     sample_conditions)
    rescale!(sample_reg_mat, lambda_X)
    sample_reg_mats = PMRegMat[copy(sample_reg_mat) for _=1:K]

    internal_sample_batch_dict = update_sample_batch_dict(sample_batch_dict,
                                                          sample_ids,
                                                          internal_samples,
                                                          internal_sample_to_idx)
    internal_sample_idx = Int[internal_sample_to_idx[s] for s in sample_ids]


    # Construct the pathway-based feature regularizer matrices (for Y)
    # and the assay-based regularizer matrix (for mu, sigma)
    feature_reg_mats, 
    assay_reg_mat, 
    internal_features, 
    internal_feat_to_idx = assemble_feature_reg_mats(pathway_sif_data, 
                                                     feature_genes, 
                                                     feature_assays)
    for mat in feature_reg_mats
        rescale!(mat, lambda_Y)
    end
    rescale!(assay_reg_mat, lambda_Y)

    internal_feature_genes = String[get_gene(feat) for feat in internal_features]
    internal_feature_losses = String[get_loss(feat) for feat in internal_features]
    internal_feature_assays = String[get_assay(feat) for feat in internal_features]

    orig_features = collect(zip(feature_genes, feature_assays))
    internal_feature_set = Set(keys(internal_feat_to_idx))
    int_feat_to_idx = feat -> (feat in internal_feature_set) ? internal_feat_to_idx[feat] : -1
    internal_feature_idx = map(int_feat_to_idx, orig_features) 

    # Construct MatFacModel
    matfac = BatchMatFacModel(sample_reg_mats, feature_reg_mats, 
                              assay_reg_mat, assay_reg_mat,
                              internal_sample_batch_dict, internal_feature_assays,
                              internal_feature_losses)


    ## TODO: still need most of these things
    model = MultiomicModel(matfac, sample_ids, sample_conditions, 
                           internal_sample_idx, internal_samples, 
                           feature_genes, feature_assays,
                           internal_feature_idx, 
                           internal_feature_genes, internal_feature_assays)

    return model
end





