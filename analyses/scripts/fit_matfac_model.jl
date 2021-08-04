

using PathwayMultiomics


function main(args)
   
    omic_hdf_filename = args[1]
    sif_filenames = args[2:end]

    features = get_omic_feature_names(omic_hdf_filename)

    sample_names = get_omic_instances(omic_hdf_filename)
    sample_groups = get_omic_groups(omic_hdf_filename)
    model = assemble_model_from_sifs(sif_filenames, nothing, features, sample_names, sample_groups)

    save_hdf("test_model.hdf", model)
    reloaded = load_hdf("test_model.hdf")

end


main(ARGS)
