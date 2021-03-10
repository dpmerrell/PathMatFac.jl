
include("factorize.jl")


function load_trained_factors(factor_hdf)

    feat_factor = h5open(factor_hdf, "r") do f
        read(f, "feature_factor")
    end
    pat_factor = h5open(factor_hdf, "r") do f
        read(f, "patient_factor")
    end
    features = h5open(factor_hdf, "r") do f
        read(f, "features")
    end
    patients = h5open(factor_hdf, "r") do f
        read(f, "patients")
    end

    println("FEATURE FACTOR: ", size(feat_factor))
    println("PATIENT FACTOR: ", size(pat_factor))
    println("FEATURES: ", size(features))
    println("PATIENTS: ", size(patients))

    return permutedims(feat_factor), permutedims(pat_factor), features, patients

end


function load_test_set(omic_hdf, test_idx)

    # Get the full vector of omic feature names
    test_features = h5open(omic_hdf, "r") do f
        read(f, "index")
    end

    # Get the data from the HDF file, and load the 
    # relevant entries into the correct rows of the test-set data
    all_data = h5open(omic_hdf, "r") do f
        read(f, "data")
    end
    test_set = all_data[test_idx, :]
      
    # Also: get the patient ids and cancer types
    #       for the test set  
    test_patients = h5open(omic_hdf, "r") do f
        read(f, "columns")
    end
    test_patients = test_patients[test_idx]
    
    test_ctypes = h5open(omic_hdf, "r") do f
        read(f, "cancer_types")
    end
    test_ctypes = test_ctypes[test_idx]

    # Return the test data, patient ids, and cancer types
    return test_set, test_features, test_patients, test_ctypes

end


function reconstruct_rrglrm(feat_factor, pat_factor, feat_ids, pat_ids)

    feat_losses = Loss[feature_to_loss(feat) for feat in feat_ids]

    rrglrm = RRGLRM(feat_factor, pat_factor, feat_ids, feat_losses, pat_ids)

    return rrglrm 
end


function save_transformed(output_hdf, transformed_data, 
                                      transformed_patients)

    h5open(output_hdf, "w") do f
        write(f, "transformed_data", transformed_data)
        write(f, "patients", convert(Vector{String}, transformed_patients))
    end
end


function main(args)

    trained_hdf = args[1]
    omic_hdf = args[2]
    train_test_json = args[3]
    output_hdf = args[4]

    # Load the results of training
    feature_factor, 
    patient_factor, 
    train_features, 
    train_patients = load_trained_factors(trained_hdf)

    # reconstruct the model 
    rrglrm = reconstruct_rrglrm(feature_factor, patient_factor, 
                                train_features, train_patients)

    # Get the test set
    test_idx = JSON.Parser.parsefile(train_test_json)["test"]
    test_idx = convert(Vector{Int64}, test_idx)
    
    test_set, test_features, test_patients, test_ctypes = load_test_set(omic_hdf, test_idx)
    println("TEST_SET", size(test_set))
    println("TEST_FEATURES", size(test_features))
    println("TEST_PATIENTS", size(test_patients))
    println("TEST_CTYPES", size(test_ctypes))

    # log-transform features, as appropriate
    log_features, std_features = get_transformations(test_features)
    test_set[:, log_features] = log.(test_set[:, log_features] .+ 10.0) 

    # Transform the test set
    transformed_data, 
    transformed_patients,
    transformed_features, ch = transform(rrglrm, test_set, test_features,
                                                 test_patients, test_ctypes)


    # Save results to an HDF file
    save_transformed(output_hdf, transformed_data, transformed_patients)
    

end


main(ARGS)


