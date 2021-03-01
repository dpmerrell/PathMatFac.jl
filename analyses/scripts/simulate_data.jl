
using JSON
using HDF5
using LinearAlgebra
using PathwayMultiomics
import PathwayMultiomics: get_all_proteins, pathways_to_matrices, hierarchy_to_matrix
import Base: isdigit

sim_omic_types = DEFAULT_DATA_TYPES


function get_all_omic_features(pwy_dict)

    all_proteins = sort(collect(get_all_proteins(pwy_dict))) 
    
    all_omic_features = String[string(prot, "_", omic) for prot in all_proteins for omic in sim_omic_types]

    return all_omic_features
end


function populate_featuremap_sim!(featuremap, features)

    for (idx, feat) in enumerate(features)
        push!(featuremap[feat], idx)
    end

    return featuremap
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



function generate_feature_factor(pwy_sifs)

    pwy_dict, empty_featuremap = load_pathways(pwy_sifs, sim_omic_types)

    omic_feature_vec = get_all_omic_features(pwy_dict) 

    populate_featuremap_sim!(empty_featuremap, omic_feature_vec)

    matrices, all_features = pathways_to_matrices(pwy_dict, empty_featuremap)
  
    pwy_names = sort(collect(keys(matrices)))

    mat_vec = [mat for (name, mat) in matrices]

    X = generate_factor(mat_vec)
 
    return X, all_features, pwy_names 
end


isdigit(s::AbstractString) = reduce((&), map(isdigit, collect(s)))


function filter_hidden_features(X, all_features)

    kept_idx = Int64[]
    omic_type_set = Set(sim_omic_types)
    
    for (i, feat) in enumerate(all_features)
        suffix = split(feat, "_")[end]
        if isdigit(suffix)
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

function sample_normal(means)
    return randn(length(means)) .+ means
end

function assign_distributions(feature_names)

    samplers = []

    for fname in feature_names
        omic_type = split(fname, "_")[end-1]
        if omic_type == "mutation"
            push!(samplers, sample_bernoulli)
        else
            push!(samplers, sample_normal)
        end
    end
    return samplers 
end


function generate_data_matrix(X, Y, sample_funcs)

    # Multiply the factors
    mu = X * transpose(Y)

    # Sample data as prescribed by the 
    # sampling functions, informed by the matrix
    # product
    for row in 1:length(sample_funcs)
        mu[row,:] = sample_funcs[row](mu[row,:])
    end    

    return mu
    
end


function save_results(output_hdf, A, X, Y, feature_vec, patient_vec, patient_hierarchy)

    patient_to_ctype = Dict(pat => ctype for (ctype, pat_vec) in patient_hierarchy for pat in pat_vec)
    ctype_vec = String[patient_to_ctype[pat] for pat in patient_vec]

    # Write to the HDF file
    h5open(output_hdf, "w") do file
        # Write factors and feature list
        write(file, "X", X)
        write(file, "Y", Y)
        write(file, "index", convert(Vector{String}, feature_vec))
        write(file, "columns", convert(Vector{String}, patient_vec)) 
        write(file, "cancer_types", convert(Vector{String}, ctype_vec))
        write(file, "data", A)
    end
end


function main(args)

    pwy_sifs = args[1:end-2]
    patient_json = args[end-1]
    output_hdf = args[end]

    # Generate the "feature" factor (m x k)
    X, all_features, pwy_names = generate_feature_factor(pwy_sifs)

    # Generate the "patient" factor (n x k)
    k = length(pwy_sifs)
    patient_hierarchy = JSON.Parser.parsefile(patient_json)
    Y, all_patients = generate_patient_factor(patient_hierarchy, k)

    # filter out the hidden features and patients
    X, kept_features = filter_hidden_features(X, all_features)
    Y, kept_patients = filter_hidden_patients(Y, all_patients)

    # Assign link functions and sampling functions to features
    # (i.e., consistent with probabilistic assumptions)
    sample_funcs = assign_distributions(kept_features)

    # Generate the data matrix (m x n)
    A = generate_data_matrix(X, Y, sample_funcs)

    println("X:")
    println(typeof(X))
    println(size(X))
    
    println("Y:")
    println(typeof(Y))
    println(size(Y))
    # Write to HDF
    save_results(output_hdf, permutedims(A), permutedims(X), permutedims(Y), kept_features, kept_patients, patient_hierarchy)

end


main(ARGS)

 
