
include("script_util.jl")


using PathwayMultiomics
using JSON
using Statistics
using ScikitLearnBase
using Profile
using ProfileSVG



function barcode_to_batch(barcode::String)

    if barcode == ""
        return ""
    end

    terms = split(barcode,"-")
    n_terms = length(terms)

    return join(terms[(n_terms-1):n_terms], "-")
end


function barcodes_to_batches(barcode_dict::Dict{String,Vector{String}})
    return Dict{String,Vector{String}}(k=>map(barcode_to_batch, v) for (k,v) in barcode_dict)
end


function main(args)
   
    omic_hdf_filename = args[1]
    pwy_json = args[2]
    out_hdf = args[3]

    opts = Dict(:max_epochs => Inf, 
                :rel_tol =>1e-8, 
                :lambda_X =>0.0, 
                :lambda_Y =>0.0,
                :lr => 0.05,
                :capacity => Int(5e7),
                :verbose => true
               )
    if length(args) > 3
        parse_opts!(opts, args[4:end])
    end

    println("OPTS:")
    println(opts)
    lambda_X = pop!(opts, :lambda_X)
    lambda_Y = pop!(opts, :lambda_Y)

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
                           batch_dict,
                           feature_genes, feature_assays;
                           lambda_X=lambda_X, 
                           lambda_Y=lambda_Y)


    start_time = time()
    fit!(model, omic_data; opts...)
    end_time = time()

    println("ELAPSED TIME (s):")
    println(end_time - start_time)

    save_hdf(model, out_hdf)

end


main(ARGS)
