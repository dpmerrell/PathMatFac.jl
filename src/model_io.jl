

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


#export save_hdf, load_hdf
#
#
#function save_hdf(file_path::String, model::MultiomicModel; save_omic_matrix::Bool=false)
#    h5open(file_path, "w") do file
#        to_hdf(file, "", model; save_omic_matrix=save_omic_matrix)
#    end
#end
#
#function load_hdf(file_path::String; load_omic_matrix::Bool=false)
#    model = h5open(file_path, "r") do file
#        multiomicmodel_from_hdf(file, ""; load_omic_matrix=load_omic_matrix)
#    end
#    return model
#end
#
#
#
#######################################
## MultiomicModel
#######################################
#
#
#function tuple_to_str(tuple)
#    return join(tuple, "::")
#end
#
#function str_to_tuple(str)
#    return tuple(split(str,"::")...)
#end
#
#
#function to_hdf(hdf_file, path::String, model::MultiomicModel; save_omic_matrix::Bool=false)
#
#    GPUMatFac.to_hdf(hdf_file, string(path, "/matfac"), model.matfac)
#    
#    write(hdf_file, string(path, "/original_genes"), model.original_genes)
#    write(hdf_file, string(path, "/original_assays"), model.original_assays)
#    write(hdf_file, string(path, "/augmented_genes"), model.augmented_genes)
#    write(hdf_file, string(path, "/augmented_assays"), model.augmented_assays)
#    
#    string_feat_to_idx = Dict([tuple_to_str(k) => v for (k,v) in model.feature_to_idx])
#    to_hdf(hdf_file, string(path, "/feature_to_idx"), string_feat_to_idx)
#
#    write(hdf_file, string(path, "/original_samples"), model.original_samples)
#    write(hdf_file, string(path, "/original_groups"), model.original_groups)
#    write(hdf_file, string(path, "/augmented_samples"), model.augmented_samples)
#    
#    to_hdf(hdf_file, string(path, "/sample_to_idx"), model.sample_to_idx)
#
#    if save_omic_matrix
#        write(hdf_file, string(path, "/omic_matrix"), model.omic_matrix)
#    end
#    
#    if model.sample_covariates != nothing
#        write(hdf_file, string(path, "/sample_covariates"), model.sample_covariates)
#    end
#
#    return
#end
#
#
#function multiomicmodel_from_hdf(hdf_file, path::String; load_omic_matrix::Bool=false)
#    
#    matfac = GPUMatFac.matfac_from_hdf(hdf_file, string(path, "/matfac"))
#    
#    original_genes = hdf_file[string(path, "/original_genes")][:]
#    original_assays = hdf_file[string(path, "/original_assays")][:]
#    augmented_genes = hdf_file[string(path, "/augmented_genes")][:]
#    augmented_assays = hdf_file[string(path, "/augmented_assays")][:]
#    str_feat_to_idx = dictionary_from_hdf(hdf_file, string(path, "/feature_to_idx"))
#    feat_to_idx = Dict([str_to_tuple(k)=>v for (k,v) in str_feat_to_idx])
#
#    original_samples= hdf_file[string(path, "/original_samples")][:]
#    original_groups = hdf_file[string(path, "/original_groups")][:]
#    augmented_samples = hdf_file[string(path, "/augmented_samples")][:]
#    sample_to_idx = dictionary_from_hdf(hdf_file, string(path, "/sample_to_idx"))
#
#    if load_omic_matrix
#        if in("omic_matrix", keys(hdf_file[path]))
#            omic_matrix = hdf_file[string(path, "/omic_matrix")][:,:]
#        else
#            @warn "No omic_matrix saved in HDF file"
#        end
#    else
#        omic_matrix = nothing
#    end
#
#    if in("sample_covariates", keys(hdf_file[path]))
#        sample_covariates = hdf_file[string(path,"/sample_covariates")][:,:]
#    else
#        sample_covariates = nothing
#    end
#
#    return MultiomicModel(matfac, original_genes, original_assays,
#                                  augmented_genes, augmented_assays,
#                                  feat_to_idx,
#                                  original_samples, original_groups,
#                                  augmented_samples,
#                                  sample_to_idx,
#                                  omic_matrix, sample_covariates)
#end
#
############################################
## Weighted edge list digraph representation
############################################
#
#function edgelist_to_hdf(hdf_file, path::String, edgelist::Vector)
#    u_vec = [edge[1] for edge in edgelist]
#    v_vec = [edge[2] for edge in edgelist]
#    w_vec = [edge[3] for edge in edgelist]
#    write(hdf_file, string(path, "/u"), u_vec)
#    write(hdf_file, string(path, "/v"), v_vec)
#    write(hdf_file, string(path, "/w"), w_vec)
#    return
#end
#
#function edgelist_from_hdf(hdf_file, path::String)
#    u_vec = hdf_file[string(path, "/u")][:]
#    v_vec = hdf_file[string(path, "/v")][:]
#    w_vec = hdf_file[string(path, "/w")][:]
#    return [[u, v_vec[i], w_vec[i]] for (i,u) in enumerate(u_vec)]
#end
#
########################################
## Dictionary
########################################
#
#function to_hdf(hdf_file, path::String, d::Dict)
#    pp = collect(d)
#    k_vec = [first(p) for p in pp]
#    v_vec = [last(p) for p in pp]
#    write(hdf_file, string(path,"/keys"), k_vec)
#    write(hdf_file, string(path,"/values"), v_vec)
#    return
#end
#
#function dictionary_from_hdf(hdf_file, path::String)
#    k_vec = hdf_file[string(path,"/keys")][:]
#    v_vec = hdf_file[string(path,"/values")][:]
#    return Dict(zip(k_vec,v_vec))
#end



