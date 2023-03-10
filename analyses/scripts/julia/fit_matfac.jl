# fit_matfac.jl
#

using PathwayMultiomics
using CUDA
using JSON
using Flux

include("script_util.jl")

PM = PathwayMultiomics


function load_omic_data(omic_hdf, omic_types)

    # Filter by omic type
    feature_assays = h5read(omic_hdf, "omic_data/feature_assays")
    omic_set = Set(omic_types)
    kept_feature_idx = map(a->in(a, omic_set), feature_assays)

    # Feature assays and genes    
    feature_assays = feature_assays[kept_feature_idx] 
    feature_genes = h5read(omic_hdf, "omic_data/feature_genes")[kept_feature_idx]
   
    # Omic data matrix 
    omic_data = h5read(omic_hdf, "omic_data/data")
    omic_data = omic_data[:,kept_feature_idx]

    # Sample ids and conditions
    sample_ids = h5read(omic_hdf, "omic_data/instances")
    sample_conditions = h5read(omic_hdf, "omic_data/instance_groups")

    return omic_data, sample_ids, sample_conditions, feature_genes, feature_assays
end


function load_batches(omic_hdf, omic_types)

    barcode_data = h5read(omic_hdf, "barcodes/data")
    batch_columns = h5read(omic_hdf, "barcodes/features")
    
    assay_to_col = Dict(a=>i for (i,a) in enumerate(batch_columns)) 
    
    result = Dict()
    for a in omic_types
        result[a] = map(barcode_to_batch, barcode_data[:,assay_to_col[a]])
    end

    return result
end


function main(args)

    #################################################
    ## PARSE COMMAND LINE ARGUMENTS
    #################################################   

    omic_hdf = args[1]
    pathway_json = args[2]
    fitted_bson = args[3]
    transformed_hdf = args[4] 

    cli_opts = Dict()
    if length(args) > 4
        cli_opts = parse_opts(args[5:end])
    end

    script_opts = Dict{Symbol,Any}(:configuration => "fsard", # {fsard, ard, graph, basic}
                            :use_batch => true,
                            :use_conditions => true,
                            :history_json => nothing,
                            :omic_types => "mrnaseq,methylation,cna,mutation",
                            :use_gpu => true,
                            :var_filter => 0.05
                            )

    update_opts!(script_opts, cli_opts)

    model_kwargs = Dict{Symbol,Any}(:K=>20,
                             :sample_ids => nothing, 
                             :sample_conditions => nothing,
                             :feature_ids => nothing, 
                             :feature_views => nothing,
                             :feature_distributions => nothing,
                             :batch_dict => nothing,
                             :sample_graphs => nothing,
                             :feature_sets => nothing,
                             :feature_graphs => nothing,
                             :lambda_X_l2 => nothing,
                             :lambda_X_condition => 1.0,
                             :lambda_X_graph => 1.0, 
                             :lambda_Y_l1 => nothing,
                             :lambda_Y_selective_l1 => nothing,
                             :lambda_Y_graph => nothing,
                             :lambda_layer => 1.0,
                             :Y_ard => false,
                             :Y_fsard => false
                             )
    update_opts!(model_kwargs, cli_opts)
 
    fit_kwargs = Dict{Symbol,Any}(:capacity => Int(10e8),
                           :verbosity => 1, 
                           :lr => 0.1,
                           :max_epochs => 1000,
                           :fit_reg_weight => "EB",
                           :lambda_max => 1.0, 
                           :n_lambda => 8,
                           :lambda_min => 1e-6,
                           :validation_frac => 0.2,
                           :fsard_max_iter => 10,
                           :fsard_max_A_iter => 1000,
                           :fsard_n_lambda => 20,
                           :fsard_lambda_atol => 1e-2,
                           :fsard_frac_atol => 0.1,
                           :fsard_A_prior_frac => 0.7,
                           :fsard_term_iter => 10,
                           :fsard_term_rtol => 1e-5,
                           :keep_history => false,
                           :verbosity => 1
                             )
    update_opts!(fit_kwargs, cli_opts)

    #################################################
    # LOAD DATA
    #################################################   
    println("LOADING DATA")

    omic_types = split(script_opts[:omic_types], ",")
 
    D, 
    sample_ids, sample_conditions, 
    feature_genes, feature_assays = load_omic_data(omic_hdf, omic_types)

    filter_idx = var_filter(D, feature_assays, script_opts[:var_filter])
    D, feature_genes, feature_assays = map(x->apply_idx_filter(x, filter_idx), [D, feature_genes, feature_assays])

    println("DATA:")
    println(size(D))

    if script_opts[:use_conditions]
        model_kwargs[:sample_conditions] = sample_conditions 
    end

    if script_opts[:use_batch]
        batch_dict = load_batches(omic_hdf, omic_types)
        model_kwargs[:batch_dict] = batch_dict 
    end

    pwy_sif_data = JSON.parsefile(pathway_json)
    pwy_edgelists = pwy_sif_data["pathways"]

    #################################################
    # PREP INPUTS
    #################################################
    println("PREPARING MODEL INPUTS")

    model_kwargs[:feature_views] = feature_assays
    model_kwargs[:feature_distributions] = map(a -> DISTRIBUTION_MAP[a], feature_assays) 

    if script_opts[:configuration] == "fsard"
        feature_ids = map(p -> join(p,"_"), zip(feature_genes, feature_assays))
        feature_sets, new_feature_ids = prep_pathway_featuresets(pwy_edgelists, 
                                                                 feature_genes;
                                                                 feature_ids=feature_ids)
        model_kwargs[:feature_sets] = feature_sets
        model_kwargs[:feature_ids] = new_feature_ids
    end

    if script_opts[:configuration] == "graph"
        feature_dogmas = map(a -> DOGMA_MAP[a], feature_assays)
        feature_ids = map(p -> join(p,"_"), zip(feature_genes, feature_assays))
        feature_graphs, new_feature_ids = prep_pathway_graphs(pwy_edgelists, 
                                                              feature_genes, 
                                                              feature_dogmas;
                                                              feature_ids=feature_ids)
        model_kwargs[:feature_graphs] = feature_graphs
        model_kwargs[:feature_ids] = new_feature_ids
    end

    #################################################
    # Construct PathMatFac
    #################################################
    println("CONSTRUCTING MODEL")

    model = PathMatFacModel(D; model_kwargs...)

    #######################################################
    ## LOAD OMIC DATA
    #######################################################
    #println("Loading data...")

    ## Load the 'omic data itself
    #feature_assays = get_omic_feature_assays(omic_hdf_filename)
    #feature_genes = get_omic_feature_genes(omic_hdf_filename)

    #sample_names = get_omic_instances(omic_hdf_filename)
    #sample_conditions = get_cancer_types(omic_hdf_filename)
    #omic_data = get_omic_data(omic_hdf_filename)

    ## Obtain some column attributes
    #feature_distributions = map(x->DISTRIBUTION_MAP[x], feature_assays)
    #feature_weights = map(x->WEIGHT_MAP[x], feature_assays)
    #feature_dogmas = map(x->DOGMA_MAP[x], feature_assays)

    ## Load the batch information
    #barcodes = get_barcodes(omic_hdf_filename)
    #batch_dict = barcodes_to_batches(barcodes) 

   
    ########################################################
    ## LOAD PATHWAYS
    ######################################################## 

    ## Load pathway edgelists
    #pwy_dict = JSON.parsefile(pwy_json) 
    #pwys = pwy_dict["pathways"]
    #pwy_names = pwy_dict["names"]
    #pwy_names = convert(Vector{String}, pwy_names)
    #pwys = convert(Vector{Vector{Vector{Any}}}, pwys)

    ## Preprocess them
    #pathway_graphs, feature_ids = prep_pathway_graphs(pwys, feature_genes,
    #                                                        feature_dogmas;
    #                                                  feature_weights=feature_weights)

    ########################################################
    ## ASSEMBLE MODEL 
    ######################################################## 
    #println("Assembling model...")
    #model = PathMatFacModel(, 
    #                        sample_names, sample_conditions,
    #                        feature_genes, feature_assays,
    #                        batch_dict;
    #                        lambda_X_l2=lambda_X_l2, lambda_X_condition=lambda_X_condition, 
    #                        lambda_Y_l1=lambda_Y_l1, lambda_Y_selective_l1=lambda_Y_selective_l1,
    #                        lambda_Y_graph=lambda_Y_graph,
    #                        lambda_layer=lambda_layer)


    ########################################################
    ## SELECT A GPU (IF APPLICABLE)
    ########################################################
    ## If a gpu_status file is provided, use it to 
    ## select an unoccupied GPU
    #status_file = nothing
    #gpu_idx = nothing
    #if haskey(opts, :gpu_status_file) 
    #    status_file = pop!(opts, :gpu_status_file)
    #    gpu_idx = get_available_device(status_file=status_file)
    #    update_device_status(gpu_idx, '1'; status_file=status_file)

    #    if gpu_idx != nothing
    #        CUDA.device!(gpu_idx-1)
    #        println(string("Using CUDA device ", gpu_idx-1))
    #    end
    #end


    ########################################################
    ## FIT THE MODEL 
    ########################################################
    #try
    #    # Move to GPU (if applicable)
    #    if use_gpu == 1
    #        omic_data_d = gpu(omic_data)
    #        model_d = gpu(model)
    #        omic_data = nothing
    #        model = nothing
    #    else
    #        omic_data_d = omic_data
    #        model_d = model
    #    end

    #    # Construct the outer callback object
    #    callback = PathwayMultiomics.OuterCallback(history_json=history_json)

    #    start_time = time()
    #    PM.fit!(model_d, omic_data_d; outer_callback=callback, opts...)
    #    end_time = time()

    #    println("ELAPSED TIME (s):")
    #    println(end_time - start_time)
    #
    #    # Move model back to CPU; save to disk
    #    model = cpu(model_d)
    #catch e
    #    if status_file != nothing
    #        update_device_status(gpu_idx, '0'; status_file=status_file) 
    #    end
    #    throw(e)
    #end


    #########################################################
    ## RELEASE GPU (IF APPLICABLE)
    #########################################################
    #if status_file != nothing
    #    update_device_status(gpu_idx, '0'; status_file=status_file) 
    #end
   
 
    #########################################################
    ## SAVE RESULTS 
    #########################################################
    #save_model(model, out_bson)

end


main(ARGS)


