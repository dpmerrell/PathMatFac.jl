
include("script_util.jl")



function run_factorization(args)

    # read command line args 
    omic_hdf = args[1]
    patient_split_json = args[2]
    data_feature_json = args[3]
    data_mask_json = args[4]
    pwy_json = args[5]
    output_hdf = args[6]
    imputed_values_json = args[7]

    log_const = 6.0

    # Get the training set patient indices
    train_idx = JSON.Parser.parsefile(patient_split_json)["train"]
    train_idx = convert(Vector{Int64}, train_idx)

    # Get the mask of missing data
    data_mask = JSON.Parser.parsefile(data_mask_json)

    # Get the indices of used features 
    used_feat_idx = JSON.Parser.parsefile(data_feature_json)
    used_feat_idx = convert(Vector{Int64}, used_feat_idx)

    # Open the data
    data_patients = get_omic_patients(omic_hdf)
    data_ctypes = get_omic_ctypes(omic_hdf)
    data_features = get_omic_feature_names(omic_hdf)

    println("LOADING OMIC DATA")
    omic_data = get_omic_data(omic_hdf)
    println("MASKING MISSING VALUES")
    apply_mask!(omic_data, data_patients, data_features, data_mask)
    
    # Restrict the data to the training set/used features 
    omic_data = omic_data[used_feat_idx, train_idx]
    data_patients = data_patients[train_idx]
    data_ctypes = data_ctypes[train_idx] 
    data_features = data_features[used_feat_idx]
    println("\t",size(omic_data)) 

    println("STANDARDIZING OMIC DATA")

    # Before fitting the model: transform 
    # the data as necessary
    #data_log_idx, data_std_idx = get_transformations(data_features)
    #omic_data[data_log_idx,:] = log.(omic_data[data_log_idx,:] .+ log_const) 

    #gs = GroupStandardizer()
    #fit!(gs, transpose(omic_data[data_std_idx,:]), data_ctypes) 
    #omic_data[data_std_idx,:] .= transpose(transform(gs, 
    #                                                 transpose(omic_data[data_std_idx,:]), 
    #                                                 data_ctypes)
    #                                       )

    ## Load the pathway SIF file paths 
    #pathway_sifs = JSON.Parser.parsefile(pwy_sifs_json)
    #pathway_sifs = convert(Vector{String}, pathway_sifs)
    pathway_dict = JSON.Parser.parsefile(pwy_json)
    pathway_ls = pathway_dict["pathways"]
    pathway_names = pathway_dict["names"]
 
    # Factorize the data! 
    imputed_matrix,
    feat_factor, pat_factor, 
    extended_features, 
    extended_patients = factorize_data(omic_data, 
                                       data_features, 
                                       data_patients,
                                       data_ctypes, 
                                       pathway_ls)
    
    # Store the estimated factors
    feat_factor = permutedims(feat_factor)
    pat_factor = permutedims(pat_factor)

    println("SAVING FACTORS")
    # Write the results to an HDF file
    save_factors(feat_factor, pat_factor, extended_features, extended_patients, pathway_names, output_hdf)
    println("Saved factors to ", output_hdf)
   
    # "un-extend" the imputed data 
    orig_pat_idx, ext_pat_idx = keymatch(data_patients, extended_patients)
    orig_feat_idx, ext_feat_idx = keymatch(data_features, extended_features)
    omic_data[orig_feat_idx, orig_pat_idx] = transpose(imputed_matrix[ext_pat_idx, ext_feat_idx])

    # Un-standardize the imputed data
    #omic_data[data_std_idx, :] = transpose(inv_transform(gs, transpose(omic_data[data_std_idx,:]), data_ctypes))

    # "un-logarithm" the imputed data
    #omic_data[data_log_idx, :] = exp.(omic_data[data_log_idx, :]) .- log_const

    # store the imputed data in a dictionary,
    # and then write to a JSON
    imputed = collect_masked_values(omic_data, data_patients, data_features, data_mask)
    open(imputed_values_json, "w") do f
        JSON.print(f, imputed)
    end

end

run_factorization(ARGS)

