

import Base: write
import BatchMatFac: readtype

export save_hdf, load_hdf


###################################
# Write
###################################

function Base.write(hdf_file::Union{HDF5.File,HDF5.Group}, path::String, 
                    obj::MultiomicModel)
    
    for prop in propertynames(obj)
        write(hdf_file, string(path, "/", prop), getproperty(obj, prop))
    end
end


function save_hdf(model::MultiomicModel, file_path::String; hdf_path::String="/")
    h5open(file_path, "w") do file
        write(file, hdf_path, model)
    end
end


###################################
# Read 
###################################

function readtype(hdf_file, path::String, t::Type{MultiomicModel})

    field_values = []
    for (fn, ft) in zip(fieldnames(t), fieldtypes(t))
        push!(field_values, readtype(hdf_file, string(path, "/", fn), ft))
    end

    return t(field_values...)
end


function load_hdf(file_path::String; hdf_path::String="/")

    model = h5open(file_path, "r") do file
        readtype(file, hdf_path, MultiomicModel)
    end
    return model

end


