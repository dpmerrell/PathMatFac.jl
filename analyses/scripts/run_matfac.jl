
using PathwayMultiomics
using ScikitLearnBase
using CUDA
using JSON
using Flux

include("script_util.jl")


function main(args)
   
    omic_hdf_filename = args[1]
    pwy_json = args[2]
    out_bson = args[3]
    out_hdf = args[4]

    opts = Dict{Symbol,Any}(:max_epochs => 1000, 
                            :rel_tol =>1e-8, 
                            :lambda_X =>0.001, 
                            :max_lambda_Y => nothing,
                            :n_lambda => 8,
                            :lambda_layer =>5.0,
                            :lr => 0.07,
                            :capacity => 25000000,
                            :verbosity => 1,
                            :fit_hyperparam => true,
                            :calibrate_losses => true
                           )
    if length(args) > 3
        parse_opts!(opts, args[5:end])
    end

    println("OPTS:")
    println(opts)
    lambda_X = pop!(opts, :lambda_X)
    lambda_layer = pop!(opts, :lambda_layer)

    println("Loading data...")
    feature_genes = get_omic_feature_genes(omic_hdf_filename)
    feature_assays = get_omic_feature_assays(omic_hdf_filename)

    sample_names = get_omic_instances(omic_hdf_filename)
    sample_conditions = get_cancer_types(omic_hdf_filename)

    omic_data = get_omic_data(omic_hdf_filename)
    barcodes = get_barcodes(omic_hdf_filename)
    batch_dict = barcodes_to_batches(barcodes) 

    println("Assembling model...")
    
    pwy_dict = JSON.parsefile(pwy_json) 
    pwys = pwy_dict["pathways"]
    pwy_names = pwy_dict["names"]
    pwy_names = convert(Vector{String}, pwy_names)
    pwys = convert(Vector{Vector{Vector{Any}}}, pwys)

    model = MultiomicModel(pwys, pwy_names, 
                           sample_names, sample_conditions,
                           feature_genes, feature_assays,
                           batch_dict;
                           lambda_X=lambda_X, 
                           lambda_Y=1.0,
                           lambda_layer=lambda_layer)

    # If a gpu_status file is provided, use it to 
    # select an unoccupied GPU
    status_file = nothing
    gpu_idx = nothing
    if haskey(opts, :gpu_status_file) 
        status_file = pop!(opts, :gpu_status_file)
        gpu_idx = get_available_device(status_file=status_file)
        update_device_status(gpu_idx, '1'; status_file=status_file)

        if gpu_idx != nothing
            CUDA.device!(gpu_idx-1)
            println(string("Using CUDA device ", gpu_idx-1))
        end
    end

    try
        # Move to GPU
        omic_data_d = gpu(omic_data)
        omic_data = nothing
        model_d = gpu(model)
        model = nothing

        start_time = time()
        ScikitLearnBase.fit!(model_d, omic_data_d; opts...)
        end_time = time()

        println("ELAPSED TIME (s):")
        println(end_time - start_time)
    
        # Move model back to CPU; save to disk
        model = cpu(model_d)
    catch e
        if status_file != nothing
            update_device_status(gpu_idx, '0'; status_file=status_file) 
        end
        throw(e)
    end

    if status_file != nothing
        update_device_status(gpu_idx, '0'; status_file=status_file) 
    end
    PathwayMultiomics.save_model(model, out_bson)
    PathwayMultiomics.save_params_hdf(model, out_hdf)

end


main(ARGS)


