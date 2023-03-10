
using HDF5, StatsBase, PathwayMultiomics, JSON

PM = PathwayMultiomics


DISTRIBUTION_MAP = Dict("mrnaseq" => "normal",
                        "methylation" => "normal",
                        "mutation" => "bernoulli",
                        "cna" => "ordinal3",
                        "rppa" => "normal")

DOGMA_MAP = Dict("mrnaseq" => "mrna",
                 "methylation" => "mrna",
                 "mutation" => "dna",
                 "cna" => "dna",
                 "rppa" => "protein")

WEIGHT_MAP = Dict("mrnaseq" => 1.0,
                  "methylation" => -1.0,
                  "mutation" => -1.0,
                  "cna" => 1.0,
                  "rppa" => 1.0)


function column_variances(data::AbstractMatrix)
    nan_idx = (!isfinite).(data)
    M = size(data, 1)
    col_counts = vec(M .- sum(nan_idx, dims=1))

    data[nan_idx] .= 0
    col_sums = vec(sum(data, dims=1))
    col_means = col_sums ./ col_counts 

    col_sq_sums = vec(sum(data.*data, dims=1))
    col_sq_means = col_sq_sums ./ col_counts

    col_variances = col_sq_sums .- (col_means.*col_means)
    empty_cols = (col_counts .== 0)
    col_variances[empty_cols] .= 0

    data[nan_idx] .= NaN

    return vec(col_variances)
end


function groups_to_idxs(feature_groups)
    result = Dict()
    for (i,g) in enumerate(feature_groups)
        s = get!(result, g, Set())
        push!(s, i)
    end
    for (k,v) in result
        result[k] = sort(collect(v))
    end
    return result
end

 
function var_filter(data, feature_groups, frac)
    
    q = 1 - frac

    col_var = column_variances(data)
    gp_to_idx = groups_to_idxs(feature_groups)

    keep_idx = Set()
    for (g, idx) in gp_to_idx
        g_var = col_var[idx]
        threshold = quantile(g_var, q)
        best_gp_idx = idx[(g_var .>= threshold)]
        union!(keep_idx, best_gp_idx)
    end

    return sort(collect(keep_idx))
end

function apply_idx_filter(x::AbstractVector, idx)
    return x[idx]
end

function apply_idx_filter(x::AbstractMatrix, idx)
    return x[:,idx]
end

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

    @assert size(omic_matrix,2) == length(feature_assays)
    @assert size(omic_matrix,1) == length(instance_names)
    @assert length(instance_names) == length(instance_groups)

    h5open(output_hdf, "w") do file
        write(file, "omic_data/feature_assays", feature_assays)
        write(file, "omic_data/feature_genes", feature_genes)
        write(file, "omic_data/instances", instance_names)
        write(file, "omic_data/instance_groups", instance_groups)
        write(file, "omic_data/data", omic_matrix)
    end

end


function parse_opts(opt_list)

    opts_k = [Symbol(split(opt,"=")[1]) for opt in opt_list]
    opts_v = [join(split(opt,"=")[2:end],"=") for opt in opt_list]

    parsed_v = Any[]
    for v in opts_v
        new_v = v
        try
            new_v = parse(Int64, v)
        catch ArgumentError
            try
                new_v = parse(Float64, v)
            catch ArgumentError
                try
                    new_v = parse(Bool, v)
                catch ArgumentError
                    new_v = string(v)
                end
            end
        finally
            push!(parsed_v, new_v)
        end
    end

    opt_d = Dict([ opts_k[i] => parsed_v[i] for i=1:length(opts_k)])

    return opt_d
end

# Wherever the keys of new_opts intersect
# the keys of defaults, update the defaults
function update_opts!(defaults, new_opts)
    for k in keys(new_opts)
        if haskey(defaults, k)
            defaults[k] = new_opts[k]
        end
    end
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



