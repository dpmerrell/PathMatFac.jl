
using HDF5
using PathwayMultiomics

function main(args)

    model_hdf = args[1]
    output_hdf = args[2]

    model = load_hdf(model_hdf)

    Z = impute(model)

    h5open(output_hdf, "w") do file
        write(file, "imputed", Z)
    end
end

main(ARGS)


