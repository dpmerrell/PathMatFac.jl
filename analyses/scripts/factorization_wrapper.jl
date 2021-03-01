
using HDF5
using PathwayMultiomics
using Statistics
using JSON

tcga_omic_types = DEFAULT_DATA_TYPES 

log_transformed_data_types = ["methylation","mrnaseq"]
standardized_data_types = ["methylation", "cna", "mrnaseq", "rppa"]


"""
    Given an empty featuremap, populate it from the array 
    of features. 
"""
function populate_featuremap_tcga(featuremap, features)

    for (idx, feat) in enumerate(features)
        
        tok = split(feat, "_")
        # extract the protein names
        prot_names = split(tok[1], " ")
        
        omic_datatype = tok[end]
 
        # for each protein name
        for protein in prot_names
            k = string(protein, "_", omic_datatype)
            if k in keys(featuremap)
                push!(featuremap[k], idx)
            end
        end
    end

    return featuremap
end


function get_tcga_patient_hierarchy(omic_hdf, cancer_types)

    hierarchy = Dict()
    h5open(omic_hdf, "r") do f

        if size(cancer_types,1) == 0
            cancer_types = unique(f["cancer_types"][:])
        end
        
        ctype_vec = f["cancer_types"][:]
        patients = f["columns"][:]

        for ctype in cancer_types
            hierarchy[ctype] = patients[ctype_vec .== ctype]
        end
    end
    return hierarchy

end


function get_omic_feature_names(omic_hdf)

    idx = h5open(omic_hdf, "r") do file
        read(file, "index")
    end

    return idx 
end



function assemble_matrix(omic_hdf, feature_vec, patient_vec)

    # Map the rows of the HDF dataset
    # to the rows of the output matrix
    matrix_rows = Vector{Int}() 
    hdf_rows = Vector{Int}()
    for (i, feature) in enumerate(feature_vec)
        suffix = split(feature, "_")[end]
        idx = tryparse(Int, suffix)
        if idx != nothing
            push!(matrix_rows, i)
            push!(hdf_rows, idx)
        end
    end

    # Initialize the matrix!
    result = fill(NaN, size(feature_vec,1), size(patient_vec,1))

    h5open(omic_hdf, "r") do f

        # Ignore the "artificial" patients: 
        # i.e., the fictitious hidden nodes
        # in the patient tree.
        hdf_patients = f["columns"][:]
        real_patients = intersect(Set(patient_vec), Set(hdf_patients))
        real_patient_vec = String[pat for pat in patient_vec if pat in real_patients]

        # Map the patients to the columns 
        # of the output matrix
        patient_to_matcol = Dict(p => idx for (idx, p) in enumerate(real_patient_vec))
        matrix_cols = Int64[patient_to_matcol[pat] for pat in real_patient_vec]

        # Map the patients to the columns
        # of the HDF file
        patient_to_hdfcol = Dict(p => idx for (idx, p) in enumerate(hdf_patients))
        hdf_cols = Int64[patient_to_hdfcol[pat] for pat in real_patient_vec]

        # Finally: load the data!    
        hdf_dataset = f["data"][:,:]
        result[matrix_rows, matrix_cols] = transpose(hdf_dataset[hdf_cols, hdf_rows])
    end

    return result
end


function feature_to_loss(feature_name)
    omic_type = split(feature_name, "_")[end]
    if omic_type == "mutation"
        return LogisticLoss()
    else
        return QuadLoss()
    end
end


function get_transformations(feature_vec)
    to_log = Int[]
    to_std = Int[]
    for (i, feat) in enumerate(feature_vec)
        tok = split(feat, "_")
        if tok[end-1] in standardized_data_types
            push!(to_std, i)
        end
        if tok[end-1] in log_transformed_data_types
            push!(to_log, i)
        end
    end
    return to_log, to_std
end

nanmean(x) = mean(filter(!isnan, x))
nanmean(x, dims) = mapslices(nanmean,x;dims=dims)

nanstd(x) = std(filter(!isnan, x))
nanstd(x, dims) = mapslices(nanstd,x;dims=dims)


function group_standardize(A, std_features, patient_vec, patient_hierarchy)

    println("group-standardizing data")

    patient_to_idx = Dict( p => i for (i, p) in enumerate(patient_vec) )

    std_params = Dict()

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


function save_results(A, X, Y, feature_vec, patient_vec, feature_losses, output_hdf)

    h5open(output_hdf, "w") do file

        write(file, "A", A)
        write(file, "X", X)
        write(file, "Y", Y)

        write(file, "features", convert(Vector{String}, feature_vec))
        write(file, "patients", convert(Vector{String}, patient_vec))

        loss_strs = String[string(l) for l in feature_losses]
        write(file, "feature_losses", loss_strs)

    end

end


function main(args)

    input_json = args[1]
 
    arg_dict = JSON.Parser.parsefile(input_json) 
    omic_hdf = arg_dict["omic_hdf"]
    pathway_sifs = arg_dict["pathway_sifs"]
    pathway_sifs = String[ps for ps in pathway_sifs] 
    cancer_types = arg_dict["cancer_types"]
    output_hdf = arg_dict["output_hdf"]
    
    data_types = tcga_omic_types 
    
    # Get the full list of omic features from the TCGA HDF file 
    omic_features = get_omic_feature_names(omic_hdf)
  
    # Read in the pathways; figure out the possible
    # ways we can map omic data on to the pathways. 
    pwys, empty_featuremap = load_pathways(pathway_sifs, data_types)

    println("FEATURE MAP: ", empty_featuremap)

    # Populate the map, using our knowledge
    # of the TCGA data
    filled_featuremap = populate_featuremap_tcga(empty_featuremap, omic_features) 

    println("FEATURE MAP: ", filled_featuremap)

    # Convert the pathways into a set of graph regularizers
    ry, feature_vec = pathways_to_regularizers(pwys, filled_featuremap)

    # Inspect the TCGA HDF file to get a hierarchy
    # of cancer types and cancer patients
    patient_hierarchy = get_tcga_patient_hierarchy(omic_hdf, cancer_types)
    # translate this hierarchy (tree) into a set of graph regularizers 
    rx, patient_vec = hierarchy_to_regularizers(patient_hierarchy, length(pwys))

    # We are now ready to assemble the full matrix for our
    # factorization problem!
    A = assemble_matrix(omic_hdf, feature_vec, patient_vec)
    println("size: ", size(A))

    # Before fitting the model: transform 
    # the features as necessary
    log_features, std_features = get_transformations(feature_vec)
    A[log_features,:] = log.(A[log_features,:] .- minimum(A[log_features,:]) .+ 1.0) 
    A, std_params = group_standardize(A, std_features, patient_vec, patient_hierarchy) 
   
    # Assign loss functions to features 
    feature_losses = Loss[feature_to_loss(feat) for feat in feature_vec]

    # Get the observed indices
    obs = findall(!isnan, transpose(A))

    # Construct the GLRM problem instance
    rrglrm = RowRegGLRM(transpose(A), feature_losses, rx, ry, 2; obs=obs)

    # Solve it!
    X, Y, ch = fit!(rrglrm)

    # We frame our problem "transposed"
    # w.r.t. the LowRankModels package
    X = permutedims(X)
    Y = permutedims(Y)

    println("X: ", size(X))
    println("Y: ", size(Y))

    # Write the results to an HDF file
    save_results(A, X, Y, feature_vec, patient_vec, 
                 feature_losses, output_hdf)

    println("Saved output to ", output_hdf)
end

main(ARGS)

