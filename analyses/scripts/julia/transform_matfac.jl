
using PathwayMultiomics
using CUDA
using JSON
using Flux

include("script_util.jl")

PM = PathwayMultiomics


function load_omic_data(omic_hdf)

    # Filter by omic type
    feature_assays = h5read(omic_hdf, "omic_data/feature_assays")

    # Feature assays and genes    
    feature_genes = h5read(omic_hdf, "omic_data/feature_genes")
   
    # Omic data matrix 
    omic_data = h5read(omic_hdf, "omic_data/data")

    # Sample ids and conditions
    sample_ids = h5read(omic_hdf, "omic_data/instances")
    sample_conditions = h5read(omic_hdf, "omic_data/instance_groups")

    return omic_data, sample_ids, sample_conditions, feature_genes, feature_assays
end


function save_transformed(transformed_X, instances, instance_groups, target, out_hdf)

    prop = HDF5.FileAccessProperties() 
    HDF5.setproperties!(prop; driver=HDF5.Drivers.Core())
    f = h5open(out_hdf, "w"; fapl=prop)

    f["X"] = transformed_X
    f["instances"] = instances
    f["instance_groups"] = instance_groups
    f["target"] = target

    close(f)
end


function main(args)

    ################################################
    # PARSE COMMAND LINE ARGUMENTS
    ################################################   
    omic_hdf = args[1]
    fitted_bson = args[2]
    out_hdf = args[3]

    opts = Dict{Symbol,Any}(:max_epochs => 1000, 
                            :lr => 1.0,
                            :rel_tol => 1e-3,
                            :verbosity => 1,
                            :use_gpu => false,
                           )
    if length(args) > 3
        cli_opts = parse_opts(args[4:end])
    end
    update_opts!(opts, cli_opts)

    println("OPTS:")
    println(opts)

    max_epochs = opts[:max_epochs]
    lr = opts[:lr]
    rel_tol = opts[:rel_tol]
    verbosity = opts[:verbosity]
    use_gpu = opts[:use_gpu]

    ######################################################
    # LOAD OMIC DATA
    ######################################################
    println("Loading data...")

    # Load the 'omic data itself
    omic_data, 
    sample_ids, sample_conditions, 
    feature_genes, feature_assays = load_omic_data(omic_hdf)
    feature_ids = map((g,a) -> string(g, "_", a), feature_genes, feature_assays)

    # Obtain some column attributes
    feature_distributions = map(x->DISTRIBUTION_MAP[x], feature_assays)
  
    # Load the prediction target, if it exists 
    target = ones(size(omic_data,1))
    try
        target = h5read(omic_hdf, "target")
    catch e
        println("No 'target' field in data HDF; using dummy")
    end

    #######################################################
    # ASSEMBLE MODEL 
    ####################################################### 

    model = load_model(fitted_bson)

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
    transformed = nothing
    try
        start_time = time()
        transformed = transform(model, omic_data;
                                feature_ids=feature_ids,
                                sample_ids=sample_ids,
                                max_epochs=max_epochs,
                                use_gpu=use_gpu, 
                                lr=lr, rel_tol=rel_tol)
        end_time = time()

        println("ELAPSED TIME (s):")
        println(end_time - start_time)
    
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
    transformed_X = permutedims(transformed.matfac.X)
    println("Transformed X")
    println(size(transformed_X))

    save_transformed(transformed_X, sample_ids, sample_conditions, 
                                    target, out_hdf)

end


main(ARGS)


