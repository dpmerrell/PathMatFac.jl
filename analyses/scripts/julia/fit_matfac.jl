# fit_matfac.jl
#

using PathwayMultiomics
using CUDA
using JSON
using Flux

include("script_util.jl")

PM = PathwayMultiomics


function prepare_featuresets(all_feature_ids, all_feature_genes, all_feature_assays,
                             view_names, view_jsons)

    all_featuresets = Dict()
    all_featureset_ids = Dict()

    for (view_name, view_json) in zip(view_names, view_jsons)
        if view_json != nothing
            rel_idx = findall(all_feature_assays .== view_name)
            rel_feature_ids = all_feature_ids[rel_idx]
            rel_feature_genes = all_feature_genes[rel_idx]
            rel_feature_assays = all_feature_assays[rel_idx]
            
            view_js_d = JSON.parsefile(view_json)
            edgelists = view_js_d["pathways"]
            pwy_names = view_js_d["names"]
            
            fs_dict, _,
            fs_id_dict = prep_pathway_featuresets(edgelists, 
                                                  rel_feature_genes,
                                                  rel_feature_assays;
                                                  feature_ids=rel_feature_ids,
                                                  featureset_ids=pwy_names)

            all_featuresets[view_name] = fs_dict[view_name]
            all_featureset_ids[view_name] = fs_id_dict[view_name]
        end
    end

    return all_featuresets, all_feature_ids, all_featureset_ids
end


function print_nan_fractions(omic_data, feature_assays)

    for unq_a in unique(feature_assays)
        rel_idx = (feature_assays .== unq_a)

        rel_data = view(omic_data, :, rel_idx)
        nanfrac = sum((!isfinite).(rel_data)) / prod(size(rel_data))
        println(string("NaN fraction for ", unq_a, ": ", nanfrac))
    end

end


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

    print_nan_fractions(omic_data, feature_assays)

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
    if length(result) == 0
        result = nothing
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
    fitted_bson = args[2]
    transformed_hdf = args[3] 

    cli_opts = Dict()
    if length(args) > 3
        cli_opts = parse_opts(args[4:end])
    end

    script_opts = Dict{Symbol,Any}(:configuration => "fsard", # {fsard, ard, graph, basic}
                                   :use_batch => true,
                                   :use_conditions => true,
                                   :history_json => nothing,
                                   :mutation_pwy_json => nothing,
                                   :methylation_pwy_json => nothing,
                                   :mrnaseq_pwy_json => nothing,
                                   :cna_pwy_json => nothing,
                                   :omic_types => "mrnaseq:methylation:cna:mutation",
                                   :gpu_status_file => nothing,
                                   :use_gpu => true,
                                   :var_filter => 0.05,
                                   )

    update_opts!(script_opts, cli_opts)

    model_kwargs = Dict{Symbol,Any}(:K => 20,
                                    :sample_ids => nothing, 
                                    :sample_conditions => nothing,
                                    :feature_ids => nothing, 
                                    :feature_views => nothing,
                                    :feature_distributions => nothing,
                                    :batch_dict => nothing,
                                    :sample_graphs => nothing,
                                    :feature_sets_dict => nothing,
                                    :featureset_names => nothing,
                                    :feature_graphs => nothing,
                                    :lambda_X_l2 => nothing,
                                    :lambda_X_condition => 1.0,
                                    :lambda_X_graph => 1.0, 
                                    :lambda_Y_l2 => 1.0,
                                    :lambda_Y_selective_l1 => nothing,
                                    :lambda_Y_graph => nothing,
                                    :lambda_layer => 1.0,
                                    :Y_ard => false,
                                    :Y_fsard => false,
                                    :fsard_v0 => 0.8
                                    )
    update_opts!(model_kwargs, cli_opts)
 
    fit_kwargs = Dict{Symbol,Any}(:capacity => Int(10e8),
                                  :verbosity => 2, 
                                  :lr => 1.0,
                                  :lr_theta => 1.0,
                                  :lr_regress => 1.0,
                                  :max_epochs => 1000,
                                  :fit_reg_weight => "EB",
                                  :fit_joint => false,
                                  :fsard_max_iter => 10,
                                  :fsard_max_A_iter => 1000,
                                  :fsard_term_rtol => 1e-3,
                                  :keep_history => false,
                                  )
    update_opts!(fit_kwargs, cli_opts)

    #################################################
    # LOAD DATA
    #################################################   
    println("LOADING DATA")

    omic_types = split(script_opts[:omic_types], ":")
 
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

    #################################################
    # PREP INPUTS
    #################################################
    println("PREPARING MODEL INPUTS")

    model_kwargs[:feature_views] = feature_assays
    model_kwargs[:feature_distributions] = map(a -> DISTRIBUTION_MAP[a], feature_assays) 

    feature_ids = map((g,a) -> string(g, "_", a), feature_genes, feature_assays)
    model_kwargs[:feature_ids] = feature_ids

    if script_opts[:configuration] == "fsard"

        # Get all of the pathway JSON files
        pwy_jsons = Dict("mutation" => script_opts[:mutation_pwy_json],
                         "methylation" => script_opts[:methylation_pwy_json],
                         "mrnaseq" => script_opts[:mrnaseq_pwy_json],
                         "cna" => script_opts[:cna_pwy_json],
                         )
        used_pwy_jsons = [pwy_jsons[ot] for ot in omic_types]
 
        # Load them into a dictionary
        feature_sets_dict, 
        new_feature_ids, 
        featureset_ids_dict = prepare_featuresets(feature_ids, feature_genes, feature_assays,
                                                  omic_types, used_pwy_jsons) 
        println("featuresets_dict")
        println(feature_sets_dict)
        
        println("featureset_ids_dict")
        println(featureset_ids_dict)

        model_kwargs[:feature_sets_dict] = feature_sets_dict
        model_kwargs[:featureset_names] = featureset_ids_dict
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
    h = nothing
    try
        # Move to GPU (if applicable)
        if use_gpu
            model_d = gpu(model)
            model = nothing
        else
            model_d = model
        end
        
        start_time = time()
        h = PM.fit!(model_d; fit_kwargs...)
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

    history_json = script_opts[:history_json]
    if history_json  != nothing
        open(history_json, "w") do f
            JSON.print(f, h)
        end 
    end
end


main(ARGS)


