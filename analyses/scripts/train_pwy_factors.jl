
include("factorize.jl")



function run_factorization(args)

    # read command line args 
    omic_hdf = args[1]
    patient_split_json = args[2]
    data_feature_json = args[3]
    pwy_sifs_json = args[4]
    output_hdf = args[5]
    gp_std_json = args[6]

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

    println("LOADING OMIC DATA")
    omic_data = get_omic_data(omic_hdf)
    omic_data = omic_data[used_feat_idx, train_idx]
    println("\t",size(omic_data)) 

    println("STANDARDIZING OMIC DATA")
    # Before fitting the model: transform 
    # the features as necessary
    log_features, std_features = get_transformations(data_features)
    omic_data[log_features,:] = log.(omic_data[log_features,:] .+ 6.0) 

    gs = GroupStandardizer()
    fit!(gs, transpose(omic_data[std_features,:]), data_ctypes) 
    omic_data[std_features,:] .= transpose(transform(gs, 
                                                     transpose(omic_data[std_features,:]), 
                                                     data_ctypes)
                                           )

    println("SAVING STANDARDIZATION PARAMS: ", gp_std_json)
    # Save the group standardizer to a JSON file
    open(gp_std_json, "w") do f
        JSON.print(f, gs.std_params)
    end

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

    println("SAVING RESULTS")
    # Write the results to an HDF file
    save_results(feat_factor, pat_factor, extended_features, extended_patients, pathway_sifs, output_hdf)

    println("Saved output to ", output_hdf)
end

run_factorization(ARGS)

