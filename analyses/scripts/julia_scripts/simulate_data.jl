
include("script_util.jl")

using LinearAlgebra
import PathwayMultiomics: get_all_proteins, pathways_to_ugraphs, 
                          ugraphs_to_matrices, hierarchy_to_matrix

import Base: isdigit

sim_omic_types = DEFAULT_OMICS


function get_all_omic_features(pwy_vec)

    all_proteins = sort(collect(get_all_proteins(pwy_vec))) 
    
    all_omic_features = String[string(prot, "_", omic) for prot in all_proteins for omic in sim_omic_types]

    return all_omic_features
end


function generate_factor(prec_mats)
    k = length(prec_mats)
    m = size(prec_mats[1], 1)    
    X = fill(NaN, m, k)

    for (i, omega) in enumerate(prec_mats)

        fac = cholesky(Symmetric(omega))
        z = randn(m)
        X[:,i] .= fac.UP \ z
    end

    return X
end


function generate_patient_factor(hierarchy, k)

    matrix, all_nodes = hierarchy_to_matrix(hierarchy)
    n = size(matrix, 1)

    # Generate k samples from N(0, I)
    z = randn(n, k)

    # Transform them by Covariance^{-1/2}
    fac = cholesky(matrix)
    Y = fac.UP \ z

    return Y, all_nodes
end


function generate_feature_factor(pathways)

    pwy_vec, featuremap = load_pathways(pathways, sim_omic_types)

    omic_feature_vec = get_all_omic_features(pwy_vec) 

    featuremap = populate_featuremap_tcga(featuremap, omic_feature_vec)

    ugraphs = pathways_to_ugraphs(pwy_vec, featuremap)
    matrices, all_features = ugraphs_to_matrices(ugraphs)

    X = generate_factor(matrices)
 
    return X, all_features  
end


function filter_hidden_features(X, all_features)

    kept_idx = Int64[]
    omic_type_set = Set(sim_omic_types)
    
    for (i, feat) in enumerate(all_features)
        suffix = split(feat, "_")[end]
        if in(suffix, omic_type_set)
            push!(kept_idx, i)
        end 
    end

    return X[kept_idx, :], all_features[kept_idx]
end


function filter_hidden_patients(Y, all_patients)

    kept_idx = Int64[]

    for (i, patient) in enumerate(all_patients)
        prefix = split(patient, "-")[1]
        if prefix == "TCGA"
            push!(kept_idx, i)
        end 
    end

    return Y[kept_idx, :], all_patients[kept_idx]
end


logistic(x) = 1.0 ./ (1.0 .+ exp.(-x))

function sample_bernoulli(logits)
    return convert(Vector{Float64}, logistic(logits) .>= rand(length(logits)))
end

function sample_normal(means; sigma=0.05)
    return (sigma .* randn(length(means))) .+ means
end

function assign_distributions(feature_names)

    samplers = []

    for fname in feature_names
        omic_type = split(fname, "_")[end]
        if omic_type == "mutation"
            push!(samplers, sample_bernoulli)
        else
            push!(samplers, sample_normal)
        end
    end
    return samplers 
end


function assign_biases(feature_names)

    biases = zeros(size(feature_names, 1))
    # TODO get rid of this hard-coded laziness
    #      (chosen s.t. probability is <.001)
    v = -6.0

    for (i, feat) in enumerate(feature_names)
        omic_type = split(feat, "_")[end]
        if omic_type == "mutation"
            biases[i] = v
        end 
    end
    return biases
end


function generate_data_matrix(X, Y, sample_funcs, biases)

    K = size(X, 2)

    # Multiply the factors. We rescale
    # by 1/sqrt(K) in order to
    # control the variance of the entries of XY^T. 
    mu = (X * transpose(Y))./ sqrt(K) .+ biases

    # Sample data as prescribed by the 
    # sampling functions, informed by the matrix
    # product
    for row in 1:length(sample_funcs)
        mu[row,:] = sample_funcs[row](mu[row,:])
    end    

    return mu
    
end


function save_results(output_hdf, A, X, Y, feature_vec, patient_vec, patient_hierarchy, pwy_names)

    patient_to_ctype = Dict(pat => ctype for (ctype, pat_vec) in patient_hierarchy for pat in pat_vec)
    ctype_vec = String[patient_to_ctype[pat] for pat in patient_vec]

    pwy_names = String[p for p in pwy_names]

    # Write to the HDF file
    h5open(output_hdf, "w") do file
        # Write factors and feature list
        write(file, "feature_factor", X)
        write(file, "instance_factor", Y)
        write(file, "features", convert(Vector{String}, feature_vec))
        write(file, "instances", convert(Vector{String}, patient_vec)) 
        write(file, "cancer_types", convert(Vector{String}, ctype_vec))
        write(file, "pathways", pwy_names)
        write(file, "data", A)
    end
end


function main(args)

    pwy_json = args[1]
    patient_json = args[2]
    output_hdf = args[3]

    # Generate the "feature" factor (m x k)
    pwy_dict = JSON.Parser.parsefile(pwy_json)
    pwys = pwy_dict["pathways"]
    pwy_names = pwy_dict["names"]
    X, all_features = generate_feature_factor(pwys)

    # Generate the "patient" factor (n x k)
    k = size(X,2)
    patient_hierarchy = JSON.Parser.parsefile(patient_json)
    Y, all_patients = generate_patient_factor(patient_hierarchy, k)

    # filter out the hidden features and patients
    X, kept_features = filter_hidden_features(X, all_features)
    Y, kept_patients = filter_hidden_patients(Y, all_patients)

    # Assign link functions and sampling functions to features
    # (i.e., consistent with probabilistic assumptions)
    sample_funcs = assign_distributions(kept_features)

    # Assign biases to features
    biases = assign_biases(kept_features)

    # Generate the data matrix (m x n)
    A = generate_data_matrix(X, Y, sample_funcs, biases)

    # Write to HDF
    save_results(output_hdf, A, X, Y, kept_features, kept_patients, patient_hierarchy, pwy_names)

end


main(ARGS)

 