
using HDF5
using PathwayMultiomics

function main(args)

    model_hdf = args[1]
    output_hdf = args[2]

    model = load_hdf(model_hdf)

    Z = impute(model)

    h5open(output_hdf, "w") do file
        write(file, "omic_matrix", Z)
        write(file, "augmented_genes", model.augmented_genes)
        write(file, "augmented_assays", model.augmented_assays)
        write(file, "augmented_samples", model.augmented_samples)
    end
end

main(ARGS)


