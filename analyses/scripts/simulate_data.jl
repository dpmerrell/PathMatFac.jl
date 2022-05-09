

include("script_util.jl")

using LinearAlgebra, JSON, HDF5, PathwayMultiomics

PM = PathwayMultiomics


function reformat_batch_ids(row_batch_ids, col_batch_ids)

    valid_idx = [idx for (idx, cbi) in enumerate(col_batch_ids) if cbi != ""]
    data = hstack(row_batch_ids[valid_idx])
    cols = col_batch_ids[valid_idx] 

    data, cols
end


function save_results(output_hdf, model, D)

    # Write to the HDF file
    h5open(output_hdf, "w") do file
        # Save the omic data
        write(file, "omic_data/data", D)
        write(file, "omic_data/feature_genes", model.feature_genes[model.feature_idx])
        write(file, "omic_data/feature_assays", model.feature_assays[model.feature_idx])
        write(file, "omic_data/instances", model.sample_ids) 
        write(file, "omic_data/instance_groups", model.sample_conditions) 
   
        # Save the batch information
        #barcode_data, barcode_cols = reformat_batch_ids(model.matfac.sample_batch_ids, 
        #                                                model.matfac.feature_batch_ids)
        #write(file, "data/barcodes/data", barcode_data)
        write(file, "barcodes/data", model.matfac.sample_batch_ids)
        write(file, "barcodes/features", unique(model.matfac.feature_batch_ids))
        write(file, "barcodes/instances", model.sample_ids)

        # Save the model (params, etc.)
        write(file, "model", model)
    end
end


function parse_args(args, param_settings)

    for arg in args
        k, v = split(arg, "=")
        v = parse(Float64, v)
        param_settings[k] = v
    end
    
    return param_settings
end


function main(args)

    # Default Values for various parameters
    opts = Dict("mu_snr" => 10.0,
                "delta_snr" => 10.0,
                "theta_snr" => 10.0,
                "logistic_snr" => 10.0,
                "sample_snr" => 10.0
               )
    
    pwy_json = args[1]
    sample_json = args[2]
    feature_json = args[3]
    moment_json = args[4]
    output_hdf = args[5]

    # Parse options
    opts = parse_args(args[6:end], opts)

    ## Get the desired statistical moments of the
    ## various assay types
    #assay_moments_dict = Dict("cna" => (opts["cna_mean"],),
    #                          "mutation" => (opts["mutation_mean"],),
    #                          "methylation" => (opts["methylation_mean"],
    #                                            opts["methylation_var"]),
    #                          "mrnaseq" => (opts["mrnaseq_mean"],
    #                                        opts["mrnaseq_var"]),
    #                          "rppa" => (opts["rppa_mean"],
    #                                     opts["rppa_var"])
    #                         )

    # Load pathway data
    pwy_dict = JSON.Parser.parsefile(pwy_json)
    pwys = convert(Vector{Vector{Vector}}, pwy_dict["pathways"])
    pwy_names = convert(Vector{String}, pwy_dict["names"])

    # Load sample info
    sample_info = JSON.Parser.parsefile(sample_json)
    sample_ids = convert(Vector{String}, sample_info["sample_ids"])
    sample_conditions = convert(Vector{String}, sample_info["sample_groups"])
    sample_barcodes_dict = sample_info["barcodes"]
    sample_barcodes_dict = convert(Dict{String,Vector{String}}, sample_barcodes_dict)
    sample_batch_dict = barcodes_to_batches(sample_barcodes_dict)

    # Load feature info
    feature_info = JSON.Parser.parsefile(feature_json)
    feature_genes = convert(Vector{String}, feature_info["feature_genes"])
    feature_assays = convert(Vector{String}, feature_info["feature_assays"])

    # Load data moments
    moment_dict = JSON.Parser.parsefile(moment_json)
    assay_moments_dict = Dict{String,Tuple}(k => Tuple(v) for (k,v) in moment_dict)

    # Generate the parameters and data matrix 
    model, sim_params, D = PM.simulate_data(pwys, pwy_names,
                                            sample_ids, sample_conditions, sample_batch_dict, 
                                            feature_genes, feature_assays,
                                            assay_moments_dict;
                                            mu_snr=opts["mu_snr"],
                                            delta_snr=opts["delta_snr"],
                                            theta_snr=opts["theta_snr"],
                                            logistic_snr=opts["logistic_snr"],
                                            sample_snr=opts["sample_snr"])

    # Write to HDF
    save_results(output_hdf, model, D)
end


main(ARGS)

 
