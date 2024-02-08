


include("script_util.jl")

using LinearAlgebra, HDF5, PathMatFac

PM = PathMatFac

function safe_convert(v::Vector{Any})
    return convert(Vector{String}, v)
end

function safe_convert(v::Vector{<:Number})
    return v
end

function write_model_to_hdf(out_hdf, model::PM.PathMatFacModel)

    # We first write everything to an HDF file in memory using the 'core' driver.
    # Then we'll write it to disk when we close().
    prop = HDF5.FileAccessProperties()
    HDF5.setproperties!(prop; driver=HDF5.Drivers.Core())
    f = h5open(out_hdf, "w"; fapl=prop)

    ###################################
    # Attributes of the data
    f["feature_ids"] = safe_convert(model.feature_ids)
    f["feature_views"] = safe_convert(model.feature_views)
    f["sample_ids"] = safe_convert(model.sample_ids)
    f["sample_conditions"] = safe_convert(model.sample_conditions)
    f["data_idx"] = safe_convert(model.data_idx)

    #################################
    # Matrix factorization: factors and column params
    f["X"] = model.matfac.X
    f["Y"] = model.matfac.Y
    f["logsigma"] = model.matfac.col_transform.layers[1].logsigma
    f["mu"] =  model.matfac.col_transform.layers[3].mu

    ###############################
    # Batch effect parameters
    if isa(model.matfac.col_transform.layers[2], PM.BatchScale)
        logdelta = model.matfac.col_transform.layers[2].logdelta
        for (i, (v, cr)) in enumerate(zip(logdelta.values, logdelta.col_ranges))
            f[string("logdelta/values_", i)] = v
            f[string("logdelta/col_range_", i)] = collect(cr)
        end
    end

    if isa(model.matfac.col_transform.layers[4], PM.BatchShift)
        theta = model.matfac.col_transform.layers[4].theta
        for (i, (v, cr, rbids)) in enumerate(zip(theta.values, theta.col_ranges, theta.row_batch_ids))
         
            f[string("theta/values_", i)] = v
            f[string("theta/col_range_", i)] = collect(cr)
            f[string("theta/batch_ids_", i)] = convert(Vector{String}, rbids)
        end
    end

    ################################
    # Featureset ARD params
    if isa(model.matfac.Y_reg, PM.FeatureSetARDReg)
        for (i, (A_mat, S_mat)) in enumerate(zip(model.matfac.Y_reg.A, model.matfac.Y_reg.S))
            f[string("fsard/A/",i)] = A_mat 
            f[string("fsard/S/",i)] = convert(Matrix{Float32}, S_mat)
        end
    end

    close(f) # Should write everything to disk at once
end


function main(args)

    in_bson = args[1]
    out_hdf = args[2]

    model = load_model(in_bson)

    rm(out_hdf, force=true)
    write_model_to_hdf(out_hdf, model)

end


main(ARGS)


