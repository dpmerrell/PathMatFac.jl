

using PathwayMultiomics


function main(args)
   
    omic_hdf_filename = args[1]
    sif_filenames = args[2:end]

    println("OMIC HDF FILE")
    println(omic_hdf_filename)

    println("FEATURE NAMES")
    features = get_omic_feature_names(omic_hdf_filename)
    println(features)

    assemble_model_from_sifs(sif_filenames, nothing, features, nothing, nothing)
    #println("SIF FILENAMES")
    #println(sif_filenames)

    #pwys, featuremap = load_pathway_sifs(sif_filenames, features)

    #println("PATHWAYS")
    #println(pwys)

    #println("FILLED FEATUREMAP")
    #println(featuremap)


end


main(ARGS)
