
using PathwayMultiomics
using JSON
using ScikitLearnBase
using Profile
using ProfileSVG
using Flux

include("script_util.jl")


function main(args)
   
    omic_hdf_filename = args[1]
    pwy_json = args[2]
    out_bson = args[3]
    out_hdf = args[4]

    opts = Dict(:max_epochs => 1000, 
                :rel_tol =>1e-8, 
                :lambda_X =>0.001, 
                :lambda_Y =>0.2,
                :lambda_layer =>5.0,
                :lr => 0.07,
                :capacity => 25000000,
                :verbosity => 1
               )
    if length(args) > 3
        parse_opts!(opts, args[5:end])
    end

    println("OPTS:")
    println(opts)
    lambda_X = pop!(opts, :lambda_X)
    lambda_Y = pop!(opts, :lambda_Y)
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
                           lambda_Y=lambda_Y,
                           lambda_layer=lambda_layer)

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
    PathwayMultiomics.save_model(out_bson, model)
    save_params_hdf(out_hdf, model)

end


main(ARGS)


