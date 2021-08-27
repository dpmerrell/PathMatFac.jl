
include("script_util.jl")

import PathwayMultiomics: get_omic_types


logistic(x) = 1.0 ./ (1.0 .+ exp.(-1.0 .* x) )


function main(args)

    omic_hdf = args[1]
    output_hdf = args[2]

    # Load feature names
    feature_names = get_omic_feature_names(omic_hdf)

    # map feature names to omic types
    omic_types = get_omic_types(feature_names)

    # Load the data
    omic_matrix = get_omic_data(omic_hdf)

    for (i, ot) in enumerate(omic_types)
        if ot == "
    end

    # transform each feature, as governed by omic type
    # * CNA ought to be transformed -> [0,1]
    

    instance_names = get_omic_instance_names(omic_hdf)
    instance_groups = get_omic_instance_groups(omic_hdf)

    save_omic_data(output_hdf, feature_names, instance_names,
                   instance_groups, transformed_matrix)
end


main(ARGS)



