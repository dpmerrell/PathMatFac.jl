
using HDF5, StatsBase, PathwayMultiomics

PM = PathwayMultiomics

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

    result = Dict(k=>barcodes[:,i] for (i,k) in enumerate(feature_groups))
    return result
end

function barcode_to_batch(barcode::String)

    if barcode == ""
        return ""
    end

    terms = split(barcode,"-")
    n_terms = length(terms)

    return join(terms[(n_terms-1):n_terms], "-")
end


function barcodes_to_batches(barcode_dict::Dict{String,Vector{String}})
    return Dict{String,Vector{String}}(k=>map(barcode_to_batch, v) for (k,v) in barcode_dict)
end


function save_omic_data(output_hdf, feature_assays, feature_genes,
                        instance_names, instance_groups, omic_matrix)

    @assert size(omic_matrix,2) == length(feature_names)
    @assert size(omic_matrix,1) == length(instance_names)
    @assert length(instance_names) == length(instance_groups)

    h5open(output_hdf, "w") do file
        write(file, "feature_assays", feature_assays)
        write(file, "feature_genes", feature_genes)
        write(file, "instances", instance_names)
        write(file, "instance_groups", instance_groups)
        write(file, "data", omic_matrix)
    end

end


function parse_opts!(defaults, opt_list)

    opts_k = [Symbol(split(opt,"=")[1]) for opt in opt_list]
    opts_v = [join(split(opt,"=")[2:end],"=") for opt in opt_list]

    parsed_v = []
    for v in opts_v
        new_v = v
        try
            new_v = parse(Int64, v)
        catch ArgumentError
            try
                new_v = parse(Float64, v)
            catch ArgumentError
                new_v = string(v)
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

nanmean(x) = mean(filter(!isnan, x))
nanvar(x) = var(filter(!isnan, x))
nanmean_and_var(x) = mean_and_var(filter(!isnan, x))


######################################
# GPU management
######################################

function get_device_statuses(; status_file="gpu_status.txt")
    
    # Get current status
    f = open(status_file, "r")
    status_str = readline(f)
    close(f)

    return status_str
end

function update_device_status(device_idx::Union{Integer,Nothing}, status::Char; status_file="gpu_status.txt")
    
    if device_idx != nothing
        status_str = get_device_statuses(;status_file=status_file)

        status_vec = collect(status_str)
        status_vec[device_idx] = status

        new_str = join(status_vec)
        f = open(status_file, "w")
        write(f, new_str)
        close(f)
    end
end

function get_available_device(; status_file="gpu_status.txt")
    status_str = get_device_statuses(status_file=status_file)
    return findfirst('0', status_str)
end



