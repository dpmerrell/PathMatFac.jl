
using PathwayMultiomics
using HDF5
using JSON

tcga_omic_types = DEFAULT_OMICS 

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


function get_omic_feature_names(omic_hdf)

    idx = h5open(omic_hdf, "r") do file
        read(file, "index")
    end

    return idx 
end


function get_omic_patients(omic_hdf)

    patients = h5open(omic_hdf, "r") do file
        read(file, "columns")
    end

    return patients 
end


function get_omic_ctypes(omic_hdf)

    cancer_types = h5open(omic_hdf, "r") do file
        read(file, "cancer_types")
    end

    return cancer_types
end


function get_omic_data(omic_hdf)

    dataset = h5open(omic_hdf, "r") do file
        read(file, "data")
    end

    # Julia reads arrays from HDF files
    # in the (weird) FORTRAN order
    return permutedims(dataset)
end


function get_transformations(feature_vec)
    to_log = Int[]
    to_std = Int[]
    for (i, feat) in enumerate(feature_vec)
        tok = split(feat, "_")
        if tok[end] in standardized_data_types
            push!(to_std, i)
        end
        if tok[end] in log_transformed_data_types
            push!(to_log, i)
        end
    end
    return to_log, to_std
end


function save_results(feature_factor, patient_factor, ext_features, ext_patients, pwy_sifs, output_hdf)

    h5open(output_hdf, "w") do file

        write(file, "feature_factor", feature_factor)
        write(file, "instance_factor", patient_factor)

        write(file, "features", convert(Vector{String}, ext_features))
        write(file, "instances", convert(Vector{String}, ext_patients))
        write(file, "pathways", convert(Vector{String}, pwy_sifs))
    end

end


function construct_glrm(A, feature_ids, feature_ugraphs, patient_ids, patient_ctypes)

    # Assign loss functions to features 
    feature_losses = Loss[feature_to_loss(feat) for feat in feature_ids]

    # Construct the GLRM problem instance
    rrglrm = RRGLRM(transpose(A), feature_losses, feature_ids, 
                                  feature_ugraphs, patient_ids, patient_ctypes;
                                  offset=true, scale=true)

end


function factorize_data(omic_data, data_features, data_patients,
                        data_ctypes, pathway_sifs)
    
    println("LOADING PATHWAYS")
    # Read in the pathways; figure out the possible
    # ways we can map omic data on to the pathways. 
    pwys, empty_featuremap = load_pathways(pathway_sifs, tcga_omic_types)

    println("POPULATING FEATURE MAP")
    # Populate the map, using our knowledge
    # of the TCGA data
    filled_featuremap = populate_featuremap_tcga(empty_featuremap, data_features) 

    # Translate the pathways into undirected graphs,
    # with data features mapped into the graph at 
    # appropriate locations 
    println("TRANSLATING PATHWAYS TO UGRAPHS")
    feature_ugraphs = pathways_to_ugraphs(pwys, filled_featuremap)

    println("CONSTRUCTING GLRM")
    # Construct the GLRM problem instance
    rrglrm = construct_glrm(omic_data, data_features, feature_ugraphs,
                                       data_patients, data_ctypes) 

    # Solve it!
    fit!(rrglrm)

    return rrglrm.Y, rrglrm.X, rrglrm.feature_ids, rrglrm.instance_ids

end


