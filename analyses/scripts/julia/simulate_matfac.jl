
# simulate_matfac.jl
#

using PathMatFac
using JSON
using Flux
using Random
#using Profile, ProfileSVG

include("script_util.jl")

PM = PathMatFac


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

    return barcode_data, batch_columns, result
end

function add_missingness!(model::PM.PathMatFacModel, barcodes, barcode_cols; missingness=0.1)

    batches = map(barcode_to_batch, barcodes)
    M, N = size(model.data)

    for (j,view) in enumerate(barcode_cols)
        rel_cols = (model.feature_views .== view)
        to_remove = Int(round(M*missingness))

        unq_batches = unique(batches[:,j])
        unq_batches = unq_batches[randperm(length(unq_batches))]
        for ub in unq_batches
            rel_rows = [i for (i,b) in enumerate(batches[:,j]) if b .== ub]
            n_rows = length(rel_rows)
            n_rem = min(n_rows, to_remove)
            if n_rem < n_rows
                selected_rows = sample(rel_rows, n_rem; replace=false)
                model.data[selected_rows, rel_cols] .= NaN
            else
                model.data[rel_rows, rel_cols] .= NaN 
            end
            to_remove -= n_rem
        end
    end 
end

function save_simulated_data(output_hdf, D, instances, instance_groups, feature_genes, feature_assays,
                                            barcode_data, barcode_columns)
    prop = HDF5.FileAccessProperties() 
    HDF5.setproperties!(prop; driver=HDF5.Drivers.Core())
    f = h5open(output_hdf, "w"; fapl=prop)

    f["omic_data/data"] = D
    f["omic_data/instances"] =  instances
    f["omic_data/instance_groups"] = instance_groups
    f["omic_data/feature_genes"] = feature_genes
    f["omic_data/feature_assays"] = feature_assays
    
    f["barcodes/data"] = barcode_data
    f["barcodes/instances"] = instances
    f["barcodes/features"] = barcode_columns

    close(f)
end

function main(args)

    #################################################
    ## PARSE COMMAND LINE ARGUMENTS
    #################################################   

    omic_hdf = args[1]
    model_bson = args[2]
    simulated_hdf = args[3] 

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
                                   :var_filter => 0.05
                                   )

    update_opts!(script_opts, cli_opts)

    model_kwargs = Dict{Symbol,Any}(:K=>10,
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
                                    :Y_fsard => false
                                    )
    update_opts!(model_kwargs, cli_opts)
 
    sim_kwargs = Dict{Symbol,Any}(:S_add_corruption => 0.1,
                                  :S_remove_corruption => 0.1,
                                  :normal_noise => 0.1,
                                  :missingness => 0.1,
                                  :A_nnz => 1,
                                  :between_batch_std => 0.25,
                                  :within_batch_std => 0.05
                                  )
    update_opts!(sim_kwargs, cli_opts)

    #################################################
    # LOAD DATA
    #################################################   
    println("LOADING REAL DATA")

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

    barcode_data, barcode_columns, batch_dict = load_batches(omic_hdf, omic_types)
    if script_opts[:use_batch]
        model_kwargs[:batch_dict] = batch_dict 
    end

    #################################################
    # PREP INPUTS
    #################################################
    println("PREPARING MODEL INPUTS")

    model_kwargs[:feature_views] = deepcopy(feature_assays)
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

    #################################################
    # Construct PathMatFac
    #################################################

    model = PathMatFacModel(D; model_kwargs...)

    ##################################################
    # Simulate model parameters and data
    ##################################################
    
    PM.simulate_params!(model; sim_kwargs...) 

    PM.simulate_data!(model; sim_kwargs...)
    println("INTRODUCING MISSINGNESS")
    add_missingness!(model, barcode_data, barcode_columns; missingness=sim_kwargs[:missingness])
 
    ########################################################
    # SAVE RESULTS 
    ########################################################

    # REMEMBER TO REORDER THE COLUMNS OF DATA S.T. THEY 
    # AGREE WITH THE ORDER OF THE ORIGINAL COLUMNS
    model.data[:, model.data_idx] .= model.data
    save_simulated_data(simulated_hdf, model.data, sample_ids, sample_conditions,
                                       feature_genes, feature_assays,
                                       barcode_data, barcode_columns) 

    save_model(model, model_bson)
end


main(ARGS)


