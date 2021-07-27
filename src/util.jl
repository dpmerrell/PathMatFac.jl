
export DEFAULT_OMICS, sort_features


DEFAULT_OMIC_DTYPES = Dict("cna" => Float64,
                           "mutation" => Bool,
                           "methylation" => Float64,
                           "mrnaseq" => Float64, 
                           "rppa" => Float64
                          )
DEFAULT_OMICS = collect(keys(DEFAULT_OMIC_DTYPES))
DEFAULT_OMIC_SET = Set(DEFAULT_OMICS)

DEFAULT_OMIC_LOSSES = Dict("cna" => "logistic",
                           "mutation" => "logistic",
                           "methylation" => "quadratic",
                           "mrnaseq" => "poisson", 
                           "rppa" => "quadratic"
                          )

DEFAULT_OMIC_MAP = Dict("cna" => ["dna", 1],
                        "mutation" => ["dna", -1],
                        "methylation" => ["mrna", -1],
                        "mrnaseq" => ["mrna", 1],
                        "rppa" => ["protein", 1]
                        )


PWY_SIF_CODE = Dict("a" => "activation",
                    "b" => "abstract",
                    "c" => "compound",
                    "h" => "chemical",
                    "p" => "protein",
                    "f" => "family"
                   )



function value_to_idx(values)
    return Dict(v => idx for (idx, v) in enumerate(values))
end


function keymatch(l_keys, r_keys)

    rkey_to_idx = value_to_idx(r_keys) 
    rkeys = Set(keys(rkey_to_idx))

    l_idx = []
    r_idx = []

    for (i, lk) in enumerate(l_keys)
        if lk in rkeys
            push!(l_idx, i)
            push!(r_idx, rkey_to_idx[lk])
        end
    end

    return l_idx, r_idx
end


function get_loss(feature_name; omic_losses=DEFAULT_OMIC_LOSSES)
    omic_type = get_omic_type(feature_name)
    if omic_type in keys(omic_losses)
        return omic_losses[omic_type], omic_type, feature_name
    else
        return "", "", feature_name
    end
end


function get_omic_type(feature_name; omic_type_set=DEFAULT_OMIC_SET)
    suff = split(feature_name, "_")[end] 
    if suff in omic_type_set
        return suff
    else
        return ""
    end
end


function sort_features(feature_names; omic_losses=DEFAULT_OMIC_LOSSES)
    tuple_list = [get_loss(feat; omic_losses=omic_losses) for feat in feature_names]
    sort!(tuple_list)
    return [t[end] for t in tuple_list]
end


#using HDF5
#using JSON


#log_transformed_data_types = [] #"methylation"]
#standardized_data_types = ["methylation", "cna", "mrnaseq", "rppa"]



#function apply_mask!(dataset, instances, features, mask)
#
#    inst_to_idx = value_to_idx(instances)
#    feat_to_idx = value_to_idx(features)
#
#    for coord in mask
#        inst_idx = inst_to_idx[coord[1]]
#        feat_idx = feat_to_idx[coord[2]]
#        dataset[feat_idx, inst_idx] = NaN
#    end 
#end
#
#
#function collect_masked_values(dataset, instances, features, mask)
#    inst_to_idx = value_to_idx(instances)
#    feat_to_idx = value_to_idx(features)
#    result = fill(NaN, length(mask))
#    for (i, coord) in enumerate(mask)
#        if (coord[1] in keys(inst_to_idx)) & (coord[2] in keys(feat_to_idx))
#            result[i] = dataset[feat_to_idx[coord[2]], inst_to_idx[coord[1]]]
#        end
#    end
#    return result
#end
#
#
#function get_transformations(feature_vec)
#    to_log = Int[]
#    to_std = Int[]
#    for (i, feat) in enumerate(feature_vec)
#        tok = split(feat, "_")
#        if tok[end] in standardized_data_types
#            push!(to_std, i)
#        end
#        if tok[end] in log_transformed_data_types
#            push!(to_log, i)
#        end
#    end
#    return to_log, to_std
#end
#
#
#function save_factors(feature_factor, patient_factor, ext_features, ext_patients, pwy_sifs, output_hdf)
#
#    h5open(output_hdf, "w") do file
#        write(file, "feature_factor", feature_factor)
#        write(file, "instance_factor", patient_factor)
#
#        write(file, "features", convert(Vector{String}, ext_features))
#        write(file, "instances", convert(Vector{String}, ext_patients))
#        write(file, "pathways", convert(Vector{String}, pwy_sifs))
#    end
#
#end



