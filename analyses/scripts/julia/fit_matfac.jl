# fit_matfac.jl
#

using PathwayMultiomics
using CUDA
using JSON
using Flux
#using Profile, ProfileSVG

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
        if a in BATCHED_ASSAYS
            result[a] = map(barcode_to_batch, barcode_data[:,assay_to_col[a]])
        end
    end

    return result
end

function save_transformed(transformed_X, instances, instance_groups, target, output_hdf)

    h5write(output_hdf, "X", transformed_X)
    h5write(output_hdf, "instances", instances)
    h5write(output_hdf, "instance_groups", instance_groups)
    h5write(output_hdf, "target", target)

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
                                   :gpu_status_file => nothing,
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
                                    :lambda_Y_l2 => nothing,
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
                                  :fsard_max_iter => 10,
                                  :fsard_max_A_iter => 10000,
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

    target = ones(size(D,1))
    try
        target = h5read(omic_hdf, "target")
    catch e
        println("No 'target' field in data HDF; using dummy")
    end

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
        model_kwargs[:Y_fsard] = true
    end

    if script_opts[:configuration] == "ard"
        model_kwargs[:Y_ard] = true
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

    if script_opts[:history_json] != nothing
        fit_kwargs[:keep_history] = true
    end

    #################################################
    # Construct PathMatFac
    #################################################

    model = PathMatFacModel(D; model_kwargs...)
    println("K LATENT FACTORS: ")
    println(size(model.matfac.X, 1))
 
    #######################################################
    # SELECT A GPU (IF APPLICABLE)
    #######################################################
    # If a gpu_status file is provided, use it to 
    # select an unoccupied GPU
    status_file = nothing
    gpu_idx = nothing
    use_gpu = script_opts[:use_gpu]

    if script_opts[:gpu_status_file] != nothing
        status_file = script_opts[:gpu_status_file]
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
        if use_gpu
            model_d = gpu(model)
            model = nothing
        else
            model_d = model
        end
        
        start_time = time()
        PM.fit!(model_d; fit_kwargs...)
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
    save_model(model, fitted_bson)
    save_transformed(permutedims(model.matfac.X), 
                     sample_ids, sample_conditions, 
                     target, transformed_hdf)

end


main(ARGS)


