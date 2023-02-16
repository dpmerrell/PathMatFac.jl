
using PathwayMultiomics
using CUDA
using JSON
using Flux

include("script_util.jl")

PM = PathwayMultiomics


function filter_omic_data!(omic_data, feature_assays, to_use)

end


function main(args)

    ################################################
    # PARSE COMMAND LINE ARGUMENTS
    ################################################   
    omic_hdf_filename = args[1]
    pwy_json = args[2]
    out_bson = args[3]
    out_hdf = args[4]

    opts = Dict{Symbol,Any}(:capacity => 25000000,
                            :max_epochs => 1000, 
                            :lr => 0.25,
                            :rel_tol => 1e-6,
                            :lambda_factor => 0.1, 
                            :lambda_layer => 0.01,
                            :lambda_X_l2 => nothing,
                            :lambda_Y_l1 => nothing,
                            :l1_fraction => 0.5,
                            :fit_hyperparam => true,
                            :lambda_max => nothing,
                            :n_lambda => 8,
                            :lambda_min_ratio => 0.4,
                            :verbosity => 1,
                            :omic_types => "mrnaseq,methylation,cna,mutation",
                            :history_json => "history.json",
                            :use_gpu => 1 
                           )
    if length(args) > 4
        parse_opts!(opts, args[5:end])
    end

    println("OPTS:")
    println(opts)
   
    ################################################### 
    # Prepare the regularizer weights
    lambda_layer = pop!(opts, :lambda_layer)
    lambda_factors = pop!(opts, :lambda_factors)

    lambda_X_condition = lambda_factors
    lambda_X_l2 = pop!(opts, :lambda_X_l2)
    if lambda_X_l2 != nothing
        lambda_X_condition = nothing
    end
    
    l1_fraction = pop!(opts, :l1_fraction)
    lambda_Y_selective_l1 = lambda_factors*l1_fraction
    lambda_Y_graph = lambda_factors*(1 - l1_fraction)
    lambda_Y_l1 = pop!(opts, :lambda_Y_l1)
    if lambda_Y_l1 != nothing
        lambda_Y_graph = nothing
        lambda_Y_selective_l1 = nothing
    end 
   
    history_json = pop!(opts, :history_json)
    omic_types = split(pop!(opts, :omic_types), ",")
    use_gpu = pop!(opts, :use_gpu)

    ######################################################
    # LOAD OMIC DATA
    ######################################################
    println("Loading data...")

    # Load the 'omic data itself
    feature_assays = get_omic_feature_assays(omic_hdf_filename)
    feature_genes = get_omic_feature_genes(omic_hdf_filename)

    sample_names = get_omic_instances(omic_hdf_filename)
    sample_conditions = get_cancer_types(omic_hdf_filename)
    omic_data = get_omic_data(omic_hdf_filename)

    # Obtain some column attributes
    feature_distributions = map(x->DISTRIBUTION_MAP[x], feature_assays)
    feature_weights = map(x->WEIGHT_MAP[x], feature_assays)
    feature_dogmas = map(x->DOGMA_MAP[x], feature_assays)

    # Load the batch information
    barcodes = get_barcodes(omic_hdf_filename)
    batch_dict = barcodes_to_batches(barcodes) 

   
    #######################################################
    # LOAD PATHWAYS
    ####################################################### 

    # Load pathway edgelists
    pwy_dict = JSON.parsefile(pwy_json) 
    pwys = pwy_dict["pathways"]
    pwy_names = pwy_dict["names"]
    pwy_names = convert(Vector{String}, pwy_names)
    pwys = convert(Vector{Vector{Vector{Any}}}, pwys)

    # Preprocess them
    pathway_graphs, feature_ids = prep_pathway_graphs(pwys, feature_genes,
                                                            feature_dogmas;
                                                      feature_weights=feature_weights)

    #######################################################
    # ASSEMBLE MODEL 
    ####################################################### 
    println("Assembling model...")
    model = PathMatFacModel(, 
                            sample_names, sample_conditions,
                            feature_genes, feature_assays,
                            batch_dict;
                            lambda_X_l2=lambda_X_l2, lambda_X_condition=lambda_X_condition, 
                            lambda_Y_l1=lambda_Y_l1, lambda_Y_selective_l1=lambda_Y_selective_l1,
                            lambda_Y_graph=lambda_Y_graph,
                            lambda_layer=lambda_layer)


    #######################################################
    # SELECT A GPU (IF APPLICABLE)
    #######################################################
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


    #######################################################
    # FIT THE MODEL 
    #######################################################
    try
        # Move to GPU (if applicable)
        if use_gpu == 1
            omic_data_d = gpu(omic_data)
            model_d = gpu(model)
            omic_data = nothing
            model = nothing
        else
            omic_data_d = omic_data
            model_d = model
        end

        # Construct the outer callback object
        callback = PathwayMultiomics.OuterCallback(history_json=history_json)

        start_time = time()
        PM.fit!(model_d, omic_data_d; outer_callback=callback, opts...)
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


    ########################################################
    # RELEASE GPU (IF APPLICABLE)
    ########################################################
    if status_file != nothing
        update_device_status(gpu_idx, '0'; status_file=status_file) 
    end
   
 
    ########################################################
    # SAVE RESULTS 
    ########################################################
    save_model(model, out_bson)
    PathwayMultiomics.save_params_hdf(model, out_hdf)

end


main(ARGS)


