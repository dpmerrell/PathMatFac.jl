
using HDF5
using PathwayMultiomics


tcga_name_map = Dict( "METH" => "methylation",
                      "CNV" => "cna",
                      "MAF" => "mutation",
                      "MRNA" => "mrnaseq",
                      "PROT" => "rppa"
                     )


"""
    Given an empty featuremap, populate it from the array 
    of features. 
"""
function populate_featuremap_tcga(featuremap, features)

    for (idx, feat) in enumerate(features)
        
        tok = split(feat, "_")
        # extract the protein names
        prot_names = split(tok[1], " ")
        
        # extract & translate the data type name
        # (this is sloppy... but it isn't part of
        # the package, so I don't feel too bad
        omic_datatype = ""
        for (k,v) in tcga_name_map
            if k in tok
                omic_datatype = v
                break
            end
        end
 
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


function get_tcga_patient_hierarchy(omic_hdf)

    hierarchy = Dict()
    h5open(omic_hdf, "r") do f
        for ctype in keys(f)
            if ctype != "index"
                hierarchy[ctype] = f[ctype]["columns"][:]
            end
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

    # Map the patients to the columns 
    # of the output matrix
    patient_to_idx = Dict(p => idx for (idx, p) in enumerate(patient_vec))

    # Initialize the matrix!
    result = fill(NaN, size(feature_vec,1), size(patient_vec,1))

    # Now populate it with data
    println("POPULATING MATRIX")
    h5open(omic_hdf, "r") do f
        for ctype in keys(f)
            println("\t", ctype)
            if ctype != "index"
                for (i,pat) in enumerate(f[ctype]["columns"][:])
                    #print(ctype, " SIZE OF DATA: ", size(f[ctype]["data"]))
                    if pat in keys(patient_to_idx)
                        for (j, feat) in enumerate(hdf_rows)
                            result[matrix_rows[j],patient_to_idx[pat]] = f[ctype]["data"][i,feat]
                        end
                    end
                end
            end
        end
    end

    return result
end



function main(args)
   
    omic_hdf = args[1]
    pathway_sifs = args[2:end] 

    data_types = values(tcga_name_map)
    omic_features = get_omic_feature_names(omic_hdf)
   
    pwys, empty_featuremap = load_pathways(pathway_sifs, data_types)

    filled_featuremap = populate_featuremap_tcga(empty_featuremap, omic_features) 

    pwy_matrices, feature_idx_decoder = pathways_to_matrices(pwys, filled_featuremap)

    patient_hierarchy = get_tcga_patient_hierarchy(omic_hdf)
    patient_matrix, patient_idx_decoder = hierarchy_to_matrix(patient_hierarchy)

    #println(pwy_matrices)
    #println(feature_idx_decoder)

    #println(patient_matrix)
    #println(patient_idx_decoder)

    mat = assemble_matrix(omic_hdf, feature_idx_decoder, patient_idx_decoder)
    println(mat)
 
end


main(ARGS)


