
include("factorize.jl")



function run_factorization(args)

    # read command line args 
    omic_hdf = args[1]
    patient_split_json = args[2]
    data_feature_json = args[3]
    pwy_sifs_json = args[4]
    output_hdf = args[5]

    # Get the training set patient indices
    train_idx = JSON.Parser.parsefile(patient_split_json)["train"]
    train_idx = convert(Vector{Int64}, train_idx)

    # Get the indices of used features 
    used_feat_idx = JSON.Parser.parsefile(data_feature_json)
    used_feat_idx = convert(Vector{Int64}, used_feat_idx)

    # Open the data, and restrict it to the 
    # training set/used features 
    data_patients = get_omic_patients(omic_hdf)
    data_patients = data_patients[train_idx]
    
    data_ctypes = get_omic_ctypes(omic_hdf)
    data_ctypes = data_ctypes[train_idx] 

    data_features = get_omic_feature_names(omic_hdf)
    data_features = data_features[used_feat_idx]

    omic_data = get_omic_data(omic_hdf)
    omic_data = omic_data[used_feat_idx, train_idx]
    
    # Before fitting the model: transform 
    # the features as necessary
    log_features, std_features = get_transformations(data_features)
    #omic_data[log_features,:] = log.(omic_data[log_features,:] .- minimum(omic_data[log_features,:]) .+ 1.0) 
    omic_data[log_features,:] = log.(omic_data[log_features,:] .+ 10.0) 

    #omic_data, std_params = group_standardize(omic_data, std_features, data_patients, data_ctypes)
    #gs = GroupStandardizer() 
    #omic_data[std_features,:] = transpose(fit_transform!(gs, transpose(omic_data[std_features,:]),
    #                                                     data_ctypes[std_features]; scale=false
    #                                                    )
    #                                      )

    # Load the pathway SIF file paths 
    pathway_sifs = JSON.Parser.parsefile(pwy_sifs_json)
    pathway_sifs = convert(Vector{String}, pathway_sifs)
    
    # Factorize the data! 
    feat_factor, pat_factor, 
    extended_features, 
    extended_patients = factorize_data(omic_data, 
                                       data_features, 
                                       data_patients,
                                       data_ctypes, 
                                       pathway_sifs)
 
    feat_factor = permutedims(feat_factor)
    pat_factor = permutedims(pat_factor)

    # Write the results to an HDF file
    save_results(feat_factor, pat_factor, extended_features, extended_patients, output_hdf)

    println("Saved output to ", output_hdf)
end

run_factorization(ARGS)

