
using PathwayMultiomics
using HDF5

include("script_util.jl")

PM = PathwayMultiomics

function perm_inverse(perm)
    @assert Set(perm) == Set(collect(1:length(perm)))
    result = zero(perm)
    for i, idx in enumerate(perm)
        result[idx] = i
    end
    return result
end

function main(args)

    model_bson = args[1]
    imputed_hdf = args[2]

    model = PM.load_model(model_bson)

    data_assays = model.data_assays
    data_genes = model.data_genes
    instances = model.sample_ids
    instance_groups = model.sample_conditions

    imputed_data = model()
    feature_idx = perm_inverse(model.used_feature_idx)
    imputed_data = imputed_data[:,feature_idx]

    save_omic_data(imputed_hdf, data_assays, data_genes,
                                instances, instance_groups, imputed_data) 
end

main(ARGS)

