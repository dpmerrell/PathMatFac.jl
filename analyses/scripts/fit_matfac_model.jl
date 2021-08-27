
include("script_util.jl")

using PathwayMultiomics


function main(args)
   
    omic_hdf_filename = args[1]
    sif_filenames = args[2:end]

    println("Loading data...")
    features = get_omic_feature_names(omic_hdf_filename)

    sample_names = get_omic_instances(omic_hdf_filename)
    sample_groups = get_omic_groups(omic_hdf_filename)

    omic_data = get_omic_data(omic_hdf_filename)
    println("Assembling model...")
    model = assemble_model_from_sifs(sif_filenames, omic_data, features, sample_names, sample_groups)

    M = size(model.omic_matrix,1)
    N = size(model.omic_matrix,2)
    println("OMIC DATA: ", M, " x ", N) 

    #save_hdf("test_model.hdf", model)
    #reloaded = load_hdf("test_model.hdf")

    println("Fitting model...")
    #fit!(model; max_iter=1000, loss_iter=1, lr=0.001, rel_tol=1e-9)
    fit!(model; method="line_search", max_iter=1000, rel_tol=1e-9, inst_reg_weight=0.1, feat_reg_weight=0.1)

    save_hdf("test_model_fitted.hdf", model)

end


main(ARGS)
