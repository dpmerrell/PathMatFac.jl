
include("factorize.jl")


nanmean(x) = mean(filter(!isnan, x))
nanmean(x, dims) = mapslices(nanmean, x; dims=dims)

nanstd(x) = std(filter(!isnan, x))
nanstd(x, dims) = mapslices(nanstd, x; dims=dims)


function group_standardize(A, std_features, patient_vec, ctype_vec)

    println("group-standardizing data")

    patient_to_idx = Dict( p => i for (i, p) in enumerate(patient_vec) )

    std_params = Dict()

    patient_hierarchy = get_instance_hierarchy(patient_vec, ctype_vec)

    for (ctype, p_vec) in patient_hierarchy

        patient_idx = Int[patient_to_idx[pat] for pat in p_vec]
        
        group_mus = nanmean(A[std_features, patient_idx], 2)
        group_sigmas = nanstd(A[std_features, patient_idx], 2)

        A[std_features,patient_idx] = (A[std_features,patient_idx] .- group_mus) ./ group_sigmas

        std_params[ctype] = Dict( "sigma" => group_sigmas,
                                  "mu" => group_mus)
    end

    return (A, std_params)
end


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
    omic_data[log_features,:] = log.(omic_data[log_features,:] .- minimum(omic_data[log_features,:]) .+ 1.0) 
    omic_data, std_params = group_standardize(omic_data, std_features, data_patients, data_ctypes) 
    

    # Load the pathway SIF file paths 
    pathway_sifs = JSON.Parser.parsefile(pwy_sifs_json)
    pathway_sifs = convert(Vector{String}, pathway_sifs)
    
    # Factorize the data! 
    feat_factor, pat_factor, 
    extended_features, 
    extended_patients, parents = factorize_data(omic_data, 
                                                data_features, 
                                                data_patients,
                                                data_ctypes, 
                                                pathway_sifs)
 
    feat_factor = permutedims(feat_factor)
    pat_factor = permutedims(pat_factor)

    # Write the results to an HDF file
    save_results(feat_factor, pat_factor, extended_features, extended_patients, 
                 parents, output_hdf)

    println("Saved output to ", output_hdf)
end

run_factorization(ARGS)

