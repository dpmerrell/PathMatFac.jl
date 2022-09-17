
import BSON

export save_model, load_model

################################
# Save/load BSON
################################
function save_model(model::MultiomicModel, filename)
    BSON.@save filename model
end

function load_model(filename::String)
    d = BSON.load(filename, @__MODULE__) 
    return d[:model]
end



####################################
# Save parameter HDF
####################################

function Base.write(f::HDF5.File, path::AbstractString, obj::Union{BatchArray, Tuple, UnitRange})
    for pname in propertynames(obj)
        x = getproperty(obj,pname)
        write(f, string(path, "/", pname), x)
    end
end


function save_params_hdf(model::MultiomicModel, hdf_filename)

    h5open(hdf_filename, "w") do f
        write(f, "X", model.matfac.X)
        write(f, "sample_ids", model.sample_ids)
        write(f, "sample_conditions", model.sample_conditions)

        write(f, "Y", model.matfac.Y)
        write(f, "data_genes", model.data_genes)
        write(f, "data_assays", model.data_assays)
        write(f, "used_feature_idx", model.used_feature_idx)

        write(f, "pathway_names", model.pathway_names)

        write(f, "mu", model.matfac.col_transform.cshift.mu)
        write(f, "logsigma", model.matfac.col_transform.cscale.logsigma)
        write(f, "theta", model.matfac.col_transform.bshift.theta)
        write(f, "logdelta", model.matfac.col_transform.bscale.logdelta)
    end

end


