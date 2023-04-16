


include("script_util.jl")

using LinearAlgebra, HDF5, PathwayMultiomics

PM = PathwayMultiomics

function safe_convert(v::Vector{Any})
    return convert(Vector{String}, v)
end

function safe_convert(v::Vector{<:Number})
    return v
end

function write_model_to_hdf(out_hdf, model::PM.PathMatFacModel)

    ###################################
    # Attributes of the data
    h5write(out_hdf, "feature_ids", safe_convert(model.feature_ids)) 
    h5write(out_hdf, "feature_views", safe_convert(model.feature_views))
    h5write(out_hdf, "sample_ids", safe_convert(model.sample_ids))
    h5write(out_hdf, "sample_conditions", safe_convert(model.sample_conditions))
    h5write(out_hdf, "data_idx", safe_convert(model.data_idx))

    #################################
    # Matrix factorization: factors and column params
    h5write(out_hdf, "X", model.matfac.X)
    h5write(out_hdf, "Y", model.matfac.Y)
    h5write(out_hdf, "logsigma", model.matfac.col_transform.layers[1].logsigma)
    h5write(out_hdf, "mu", model.matfac.col_transform.layers[3].mu)

    ###############################
    # Batch effect parameters
    if isa(model.matfac.col_transform.layers[2], PM.BatchScale)
        logdelta = model.matfac.col_transform.layers[2].logdelta
        for (i, (v, cr)) in enumerate(zip(logdelta.values, logdelta.col_ranges))
            h5write(out_hdf, string("logdelta/values_", i), v)
            h5write(out_hdf, string("logdelta/col_range_", i), collect(cr))
        end
    end

    if isa(model.matfac.col_transform.layers[4], PM.BatchShift)
        theta = model.matfac.col_transform.layers[4].theta
        for (i, (v, cr)) in enumerate(zip(theta.values, theta.col_ranges))
            h5write(out_hdf, string("theta/values_", i), v)
            h5write(out_hdf, string("theta/col_range_", i), collect(cr))
        end
    end

    ################################
    # Featureset ARD params
    if isa(model.matfac.Y_reg, PM.FeatureSetARDReg)
        h5write(out_hdf, string("fsard/A"), model.matfac.Y_reg.A)
        #h5write(out_hdf, string("fsard/S"), convert(Matrix{Float32}, model.matfac.Y_reg.S))
    end
end


function main(args)

    in_bson = args[1]
    out_hdf = args[2]

    model = load_model(in_bson)

    rm(out_hdf, force=true)
    write_model_to_hdf(out_hdf, model)

end


main(ARGS)


