
using BatchMatFac
using PathwayMultiomics



function main(args)

    opts = Dict
    if length(args) > TODO
        extra_args = args[TODO] 
        parse_opts!(opts, extra_args)
    end

    pwy_json = args[1]
    sample_json = args[2]
    feature_json = args[3]

    # Load pathways from JSON
    pwy_dict = JSON.parsefile(pwy_json) 
    pwys = pwy_dict["pathways"]
    pwy_names = pwy_dict["names"]
    pwy_names = convert(Vector{String}, pwy_names)

    # Load sample information from JSON
    sample_dict = JSON.parsefile(sample_json)
    sample_ids = convert(Vector{String}, sample_dict["sample_ids"])
    sample_conditions = convert(Vector{String}, sample_dict["sample_conditions"])
    sample_batch_dict = convert(Dict{String,Vector{String}}, 
                                sample_dict["sample_batch_dict"])

    # Load feature information from JSON
    feature_dict = JSON.parsefile(feature_json)
    feature_genes = convert(Vector{String}, feature_dict["feature_genes"])
    feature_assays = convert(Vector{String}, feature_dict["feature_assays"])

    sim_params, D = simulate_data(pathway_sif_data, 
                                  pathway_names::Vector{String},
                                  sample_ids::Vector{String}, 
                                  sample_conditions::Vector{String},
                                  sample_batch_dict::Dict{T,Vector{U}},
                                  feature_genes::Vector{String}, 
                                  feature_assays::Vector{T}) where T where U
end

main(ARGS)

#using LinearAlgebra, JSON, HDF5
#import PathwayMultiomics: extend_pathway,
#                          get_all_proteins, 
#                          build_instance_hierarchy,
#                          hierarchy_to_matrix,
#                          add_data_nodes_to_pathway,
#                          assemble_feature_reg_mats,
#                          assemble_instance_reg_mat,
#                          DEFAULT_OMIC_MAP
#
#import Base: isdigit
#
#sim_omic_types = collect(keys(DEFAULT_OMIC_MAP))
#
#logistic(x) = 1.0 ./ (1.0 .+ exp.(-x))
#
#link_functions = Dict( "cna" => identity,
#                       "mutation" => logistic,
#                       "methylation" => identity,
#                       "mrnaseq" => identity,
#                       "rppa" => identity
#                      )
#
#
#function get_all_omic_features(pwy_vec)
#
#    all_proteins = sort(collect(get_all_proteins(pwy_vec))) 
#    
#    all_omic_features = String[string(prot, "_", omic) for prot in all_proteins for omic in sim_omic_types]
#
#    return all_omic_features
#end
#
#
#function generate_patient_factor(patients, patient_groups, k, temperature)
#
#    patient_hierarchy = build_instance_hierarchy(patients, patient_groups)
#    matrix, all_nodes = hierarchy_to_matrix(patient_hierarchy)
#    n = size(matrix, 1)
#
#    # Generate k samples from N(0, I)
#    z = randn(n, k)
#
#    # Transform them by Covariance^{-1/2}
#    fac = cholesky(matrix)
#    Y = transpose((fac.UP \ z) .* sqrt(temperature))
#
#    return Y, all_nodes
#end
#
#
#function generate_factor(prec_mats, temperature)
#    k = length(prec_mats)
#    m = size(prec_mats[1], 1)    
#    X = fill(NaN, m, k)
#
#    for (i, omega) in enumerate(prec_mats)
#
#        fac = cholesky(Symmetric(omega))
#        z = randn(m)
#        X[:,i] .= (fac.UP \ z) .* sqrt(temperature)
#    end
#
#    return transpose(X)
#end
#
#
#function generate_feature_factor(pathways, temperature)
#
#    # First, need to extend the pathways
#    extended_pwys = [extend_pathway(pwy) for pwy in pathways]
#
#    # Then, get all of the omic features from the extended pathways 
#    feature_names = get_all_omic_features(extended_pwys) 
#    
#    matrices, 
#    aug_features, _ = assemble_feature_reg_mats(pathways, feature_names) 
#
#    X = generate_factor(matrices, temperature)
# 
#    return X, aug_features
#end
#
#
#function filter_hidden_features(X, all_features)
#
#    kept_idx = Int64[]
#    omic_type_set = Set(sim_omic_types)
#    
#    for (i, feat) in enumerate(all_features)
#        suffix = split(feat, "_")[end]
#        if in(suffix, omic_type_set)
#            push!(kept_idx, i)
#        end 
#    end
#
#    return X[:, kept_idx], all_features[kept_idx]
#end
#
#
#function filter_hidden_patients(Y, all_patients)
#
#    kept_idx = Int64[]
#
#    for (i, patient) in enumerate(all_patients)
#        prefix = split(patient, "-")[1]
#        if prefix == "TCGA"
#            push!(kept_idx, i)
#        end 
#    end
#
#    return Y[:, kept_idx], all_patients[kept_idx]
#end
#
#
#function generate_biases(feature_names, param_settings) 
#
#    biases = zeros(size(feature_names, 1))
#
#    for (i, feat) in enumerate(feature_names)
#        omic_type = split(feat, "_")[end]
#
#        biases[i] = randn()*param_settings[string("b_std_",omic_type)] \
#                           + param_settings[string("b_mu_",omic_type)]
#
#    end
#    return biases
#end
#
#
#function apply_link_functions(mu, omic_types)
#
#    for (j, ot) in enumerate(omic_types)
#        mu[:,j] .= link_functions[ot](mu[:,j])
#    end
#
#    return mu
#end
#
#
#function apply_noise(A, omic_types, param_settings)
#
#    M = size(A,1)
#    for (j, ot) in enumerate(omic_types)
#        if ot == "mutation"
#            A[:,j] .= float(rand(M) .<= A[:,j])
#        else
#            A[:,j] .= A[:,j] .+ randn(M)*param_settings[string("noise_std_",ot)]
#        end
#    end
#
#    return A
#end
#
#
#function generate_data_matrix(X, Y, biases, omic_types, param_settings)
#
#    mu = (transpose(X) * Y) .+ transpose(biases)
#
#    mu = apply_link_functions(mu, omic_types)
#    mu = apply_noise(mu, omic_types, param_settings)
#
#    return mu    
#end
#
#
#function save_results(output_hdf, A, X, Y, b, instance_vec, 
#                      feature_vec, group_vec, pwy_names)
#
#    pwy_names = String[p for p in pwy_names]
#
#    # Write to the HDF file
#    h5open(output_hdf, "w") do file
#        write(file, "data", A)
#        write(file, "features", convert(Vector{String}, feature_vec))
#        write(file, "instances", convert(Vector{String}, instance_vec)) 
#        
#        write(file, "X", X)
#        write(file, "Y", Y)
#        write(file, "b", b)
#        write(file, "groups", convert(Vector{String}, group_vec))
#        write(file, "pathways", pwy_names)
#    end
#end
#
#function parse_args(args, param_settings)
#
#    for arg in args[4:end]
#        k, v = split(arg, "=")
#        v = parse(Float64, v)
#        param_settings[k] = v
#    end
#    
#    return param_settings
#end
#
#
#function main(args)
#
#    # Default Values for various parameters
#    param_settings = Dict(
#                          "X_temperature" => 2.0,
#                          "Y_temperature" => 0.5,
#                          "b_mu_mutation" => -6.0, 
#                          "b_mu_cna" => 0.0, 
#                          "b_mu_methylation" => 0.0,
#                          "b_mu_mrnaseq" => 0.0, 
#                          "b_mu_rppa" => 0.0,
#                          "b_std_mutation" => 0.5, 
#                          "b_std_cna" => 0.1, 
#                          "b_std_methylation" => 1.0,
#                          "b_std_mrnaseq" => 3.0, 
#                          "b_std_rppa" => 1.0,
#                          "b_temperature" => 1.0,
#                          "noise_std_cna" => 0.1, 
#                          "noise_std_methylation" => 1.0,
#                          "noise_std_mrnaseq" => 3.0, 
#                          "noise_std_rppa" => 1.0
#                         )
#    
#    pwy_json = args[1]
#    patient_json = args[2]
#    output_hdf = args[3]
#
#    param_settings = parse_args(args, param_settings)
#
#    pwy_dict = JSON.Parser.parsefile(pwy_json)
#    pwys = pwy_dict["pathways"]
#    pwy_names = pwy_dict["names"]
#
#    # Generate the "patient" factor (K x M)
#    K = length(pwys) 
#    patient_dict = JSON.Parser.parsefile(patient_json)
#    patients = patient_dict["instances"]
#    groups = patient_dict["groups"]
#
#    X, all_patients = generate_patient_factor(patients, groups, K, 
#                                              param_settings["X_temperature"]/K)
#    # Generate the "feature" factor (K x N)
#    Y, all_features = generate_feature_factor(pwys, param_settings["Y_temperature"]/K)
#
#
#    # filter out the hidden features and patients
#    X, kept_patients = filter_hidden_patients(X, all_patients)
#    Y, kept_features = filter_hidden_features(Y, all_features)
#
#    # Assign biases to features
#    biases = generate_biases(kept_features, param_settings)
#
#    # Generate the data matrix (m x n)
#    omic_types = [split(feat, "_")[end] for feat in kept_features] 
#    A = generate_data_matrix(X, Y, biases, omic_types, param_settings)
#
#    # Write to HDF
#    save_results(output_hdf, A, X, Y, biases, kept_patients, kept_features, groups, pwy_names)
#
#end
#
#
#main(ARGS)

 
