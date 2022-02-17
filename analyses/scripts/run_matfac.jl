
include("script_util.jl")


using PathwayMultiomics
using JSON
using Statistics
using ScikitLearnBase
using Profile
using ProfileSVG

function parse_opts!(defaults, opt_list)

    opts_k = [Symbol(split(opt,"=")[1]) for opt in opt_list]
    opts_v = [split(opt,"=")[end] for opt in opt_list]

    parsed_v = []
    for v in opts_v
        new_v = v
        try
            new_v = parse(Int64, v)
        catch ArgumentError
            try
                new_v = parse(Float64, v)
            catch ArgumentError
                new_v = v
            end
        finally
            push!(parsed_v, new_v)
        end
    end

    opt_d = Dict([ opts_k[i] => parsed_v[i] for i=1:length(opts_k)])

    for (opt_k, opt_v) in opt_d
        defaults[opt_k] = opt_v
    end

    return defaults

end


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

    opts = Dict(:max_iter => Inf, 
                :rel_tol =>1e-9, 
                :lambda_X =>0.1, 
                :lambda_Y =>0.1,
                :lr => 0.01
               )
    if length(args) > 3
        parse_opts!(opts, args[4:end])
    end

    println("OPTS:")
    println(opts)
    lambda_X = opts[:lambda_X]
    lambda_Y = opts[:lambda_Y]

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


    #println("One iteration (for precompilation)")
    #fit!(model, omic_data; verbose=true, max_epochs=1, capacity=Int(5e7))
    
    #println("Multiple iterations (for profiling)")
    #Profile.init(delay=0.01)
    start_time = time()
    fit!(model, omic_data; verbose=true, max_epochs=500, capacity=Int(5e7), lr=0.02)
    end_time = time()

    println("ELAPSED TIME (s):")
    println(end_time - start_time)

    #ProfileSVG.save("fit_profile.svg", maxframes=5000)

    save_hdf(model, out_hdf)

end


main(ARGS)
