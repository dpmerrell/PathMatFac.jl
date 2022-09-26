

include("script_util.jl")

using LinearAlgebra, JSON, HDF5, PathwayMultiomics

PM = PathwayMultiomics


function save_sim_data(output_hdf, D, sample_ids, sample_conditions,
                                      feature_genes, feature_assays,
                                      sample_batch_dict)

    # Write to the HDF file
    h5open(output_hdf, "w") do file
        # Save the omic data
        write(file, "omic_data/data", D)
        write(file, "omic_data/feature_genes", feature_genes)
        write(file, "omic_data/feature_assays", feature_assays)
        write(file, "omic_data/instances", sample_ids) 
        write(file, "omic_data/instance_groups", sample_conditions) 
   
        # Save the batch information
        feature_batch_ids = collect(keys(sample_batch_dict))
        sample_batch_ids = fill("", length(sample_ids), length(feature_batch_ids))
        for (i,fb) in enumerate(feature_batch_ids)
            sample_batch_ids[:,i] .= sample_batch_dict[fb] 
        end

        write(file, "barcodes/data", sample_batch_ids)
        write(file, "barcodes/features", feature_batch_ids)
        write(file, "barcodes/instances", sample_ids)

    end
end


function main(args)

    ## Default Values for various parameters
    opts = Dict(:total_var => 25.0,
                :snr => 999.0
               )

    pwy_json = args[1]
    sample_json = args[2]
    feature_json = args[3]
    data_hdf = args[4]
    model_hdf = args[5]

    # Parse options
    opts = parse_opts!(opts, args[6:end])
    total_var = opts[:total_var]
    snr = opts[:snr]

    noise_var = total_var/(1 + snr)
    X_var = snr*noise_var

    # Load pathway data
    println("Loading pathway data")
    pwy_dict = JSON.Parser.parsefile(pwy_json)
    pwys = convert(Vector{Vector{Vector}}, pwy_dict["pathways"])
    pwy_names = convert(Vector{String}, pwy_dict["names"])

    # Load sample info
    println("Loading sample info")
    sample_info = JSON.Parser.parsefile(sample_json)
    sample_ids = convert(Vector{String}, sample_info["sample_ids"])
    sample_conditions = convert(Vector{String}, sample_info["sample_groups"])
    sample_barcodes_dict = sample_info["barcodes"]
    sample_barcodes_dict = convert(Dict{String,Vector{String}}, sample_barcodes_dict)
    sample_batch_dict = barcodes_to_batches(sample_barcodes_dict)

    # Load feature info
    println("Loading feature info")
    feature_info = JSON.Parser.parsefile(feature_json)
    feature_genes = convert(Vector{String}, feature_info["feature_genes"])
    feature_assays = convert(Vector{String}, feature_info["feature_assays"])

    # Generate the parameters and data matrix 
    println("Simulating data")
    model, D = PM.simulate_data(pwys, pwy_names,
                                sample_ids, sample_conditions, 
                                feature_genes, feature_assays,
                                sample_batch_dict;
                                X_var=X_var, noise_var=noise_var) 

    # Write to HDF
    println("Saving data to HDF")
    save_sim_data(data_hdf, D, sample_ids, sample_conditions,
                               feature_genes, feature_assays,
                               sample_batch_dict)
    println("Saving parameters to HDF")
    PM.save_params_hdf(model, model_hdf)
end


main(ARGS)

 
