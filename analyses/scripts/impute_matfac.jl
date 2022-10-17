

include("script_util.jl")


function main(args)

    model_bson = args[1]
    imputed_hdf = args[2]

    pm_model = PM.load_model(model_bson)

    data_assays = pm_model.data_assays
    data_genes = pm_model.data_genes
    instances = pm_model.sample_ids
    instance_groups = pm_model.sample_conditions

    imputed_data = PM.impute(pm_model)
    imputed_data[:,pm_model.used_feature_idx] .= imputed_data

    save_omic_data(imputed_hdf, data_assays, data_genes,
                                instances, instance_groups, imputed_data)
 
end

main(ARGS)

