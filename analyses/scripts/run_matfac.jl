
include("script_util.jl")


using PathwayMultiomics
using JSON
using Statistics

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


function inspect_omic_data(feature_assays, omic_data)

    println("OMIC DATA: ", size(omic_data))
    properties = Dict([ot => Dict("min" => Inf, "max"=>-Inf) for ot in unique(feature_assays)])

    for (i, ot) in enumerate(feature_assays)
        properties[ot]["min"] = min(properties[ot]["min"], minimum(omic_data[:,i])) 
        properties[ot]["max"] = max(properties[ot]["max"], minimum(omic_data[:,i])) 
    end
    println(properties)
end

function main(args)
   
    omic_hdf_filename = args[1]
    pwy_json = args[2]
    out_hdf = args[3]

    opts = Dict(:method =>"adagrad", 
                :max_iter =>1000, 
                :rel_tol =>1e-9, 
                :inst_reg_weight =>0.1, 
                :feat_reg_weight =>0.1,
                :lr => 0.01
               )
    if length(args) > 3
        parse_opts!(opts, args[4:end])
    end

    println("OPTS:")
    println(opts)

    println("Loading data...")
    feature_genes = get_omic_feature_genes(omic_hdf_filename)
    feature_assays = get_omic_feature_assays(omic_hdf_filename)

    sample_names = get_omic_instances(omic_hdf_filename)
    sample_groups = get_omic_groups(omic_hdf_filename)

    omic_data = get_omic_data(omic_hdf_filename)

    inspect_omic_data(feature_assays, omic_data)

    println("Assembling model...")
    
    pwy_dict = JSON.parsefile(pwy_json) 
    pwys = pwy_dict["pathways"]

    model = assemble_model(pwys, omic_data, 
                           feature_genes, feature_assays,
                           sample_names, sample_groups)

    M = size(model.omic_matrix,1)
    N = size(model.omic_matrix,2)
    println("OMIC DATA: ", M, " x ", N) 

    #save_hdf("test_model.hdf", model)
    #reloaded = load_hdf("test_model.hdf")

    println("Fitting model...")
    fit!(model; opts...)

    save_hdf(out_hdf, model)

end


main(ARGS)
