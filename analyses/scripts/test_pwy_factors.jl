
include("factorize.jl")


function load_trained_factors(factor_hdf)

    pwy_factor = h5open(factor_hdf, "r") do f
        read(f, "X")
    end
    pat_factor = h5open(factor_hdf, "r") do f
        read(f, "Y")
    end
    features = h5open(factor_hdf, "r") do f
        read(f, "features")
    end
    patients = h5open(factor_hdf, "r") do f
        read(f, "patients")
    end
    parents = h5open(factor_hdf, "r") do f
        read(f, "parents")
    end

    return pwy_factor, pat_factor, features, patients, parents

end


function load_test_set(omic_hdf, matrix_features, test_idx)

    # Get the full vector of omic feature names
    data_features = h5open(omic_hdf, "r") do f
        read(f, "index")
    end


    # Figure out which omic features need to 
    # be extracted, and their corresponding rows
    # in the pathway factors
    feat_to_hdf_idx = Dict(feat => idx for (idx, feat) in enumerate(data_features))
    data_feature_set = Set(data_features)
    hdf_rows = Int64[]
    mat_rows = Int64[]
    for (mat_idx, mat_feat) in enumerate(matrix_features)
        if mat_feat in data_feature_set
            push!(hdf_rows, feat_to_hdf_idx[mat_feat])
            push!(mat_rows, mat_idx)
        end
    end 

    # Pre-allocate the test set data
    test_set = fill(NaN, (size(matrix_features,1), size(test_idx,1)))

    # Get the data from the HDF file, and load the 
    # relevant entries into the correct rows of the test-set data
    all_data = h5open(omic_hdf, "r") do f
        read(f, "data")
    end
    test_set[mat_rows, :] = transpose(all_data)[hdf_rows, test_idx]
      
 
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
    return test_set, test_patients, test_ctypes

end


function construct_glrm_transformer(pwy_factor, pat_factor, train_patients, 
                                    test_set, test_parents)

    r_pat = 

    r_feat = Regularizer[FixedLatentFeatureConstraint(pwy_factor[:,i]) 
                         for i=1:size(pwy_factor,2)]
    
    rrglrm = construct_glrm(test_set, r_patients, patients, )

    return rrglrm 
end


function infer_latent(rrlgrm, test_data, test_features, 
                              test_patients, test_parents)

end


function main(args)

    trained_hdf = args[1]
    omic_hdf = args[2]
    train_test_json = args[3]

    # Load the results of training
    pwy_factor, pat_factor, features, patients, parents = load_trained_factors(trained_hdf)

    # Get the test set
    test_idx = JSON.Parser.parsefile(train_test_json)["test"]
    test_idx = convert(Vector{Int64}, test_idx)
    
    test_set, test_patients, test_ctypes = load_test_set(omic_hdf, features, test_idx)
    println("TEST_SET", size(test_set))
    println("TEST_PATIENTS", size(test_patients))
    println("TEST_CTYPES", size(test_ctypes))

    # construct the RRGLRM instance
    rrglrm = construct_rrglrm(pwy_factor, pat_factor, features, patients, parents)

    #println("PWY_FACTOR: ", size(pwy_factor))
    #println("PAT_FACTOR: ", size(pat_factor))
    #println("FEATURES: ", size(features))
    #println("PATIENTS: ", size(patients))
    #println("PARENTS: ", size(parents))
    

end


main(ARGS)


