
using HDF5

function get_omic_feature_genes(omic_hdf)

    genes = h5open(omic_hdf, "r") do file
        read(file, "omic_data/feature_genes")
    end

    return genes
end

function get_omic_feature_assays(omic_hdf)

    assays = h5open(omic_hdf, "r") do file
        read(file, "omic_data/feature_assays")
    end

    return assays
end

function get_omic_instances(omic_hdf)

    patients = h5open(omic_hdf, "r") do file
        read(file, "omic_data/instances")
    end

    return patients 
end


function get_cancer_types(omic_hdf)

    cancer_types = h5open(omic_hdf, "r") do file
        read(file, "omic_data/instance_groups")
    end

    return cancer_types
end


function get_omic_data(omic_hdf)

    dataset = h5open(omic_hdf, "r") do file
        read(file, "omic_data/data")
    end

    return dataset
end

function get_barcodes(omic_hdf)

    barcodes, feature_groups = h5open(omic_hdf, "r") do file
        barcodes = read(file, "barcodes/data")
        feature_groups = read(file, "barcodes/features")
        return barcodes, feature_groups
    end

    result = Dict(k[1:(end-1)]=>barcodes[:,i] for (i,k) in enumerate(feature_groups))
    return result
end


function save_omic_data(output_hdf, feature_names, instance_names, 
                        instance_groups, omic_matrix)

    @assert size(omic_matrix,2) == length(feature_names)
    @assert size(omic_matrix,1) == length(instance_names)
    @assert length(instance_names) == length(instance_groups)

    h5open(output_hdf, "w") do file
        write(file, "features", feature_names)
        write(file, "instances", instance_names)
        write(file, "groups", instance_groups)
        write(file, "data", omic_matrix)
    end

end


function parse_opts!(defaults, opt_list)

    opts_k = [Symbol(split(opt,"=")[1]) for opt in opt_list]
    opts_v = [split(opt,"=")[end] for opt in opt_list]

    parsed_v = []
    for v in opts_v
        new_v = v
        try
            new_v = parse(Int64, v)
        catch ArgumentError
            try
                new_v = parse(Float64, v)
            catch ArgumentError
                new_v = v
            end
        finally
            push!(parsed_v, new_v)
        end
    end

    opt_d = Dict([ opts_k[i] => parsed_v[i] for i=1:length(opts_k)])

    for (opt_k, opt_v) in opt_d
        defaults[opt_k] = opt_v
    end

    return defaults

end
