

using HDF5

export save_hdf, load_hdf


function save_hdf(file_path::String, model::MultiomicModel; save_omic_matrix::Bool=false)
    h5open(file_path, "w") do file
        to_hdf(file, "", model; save_omic_matrix=save_omic_matrix)
    end
end

function load_hdf(file_path::String; load_omic_matrix::Bool=false)
    model = h5open(file_path, "r") do file
        multiomicmodel_from_hdf(file, ""; load_omic_matrix=load_omic_matrix)
    end
    return model
end



######################################
# MultiomicModel
######################################

function to_hdf(hdf_file, path::String, model::MultiomicModel; save_omic_matrix::Bool=false)

    GPUMatFac.to_hdf(hdf_file, string(path, "/matfac"), model.matfac)
    
    write(hdf_file, string(path, "/feature_genes"), model.feature_genes)
    write(hdf_file, string(path, "/feature_assays"), model.feature_assays)

    write(hdf_file, string(path, "/sample_ids"), model.sample_ids)
    write(hdf_file, string(path, "/sample_groups"), model.sample_ids)

    if save_omic_matrix
        write(hdf_file, string(path, "/omic_matrix"), model.omic_matrix)
    end

    return
end


function multiomicmodel_from_hdf(hdf_file, path::String; load_omic_matrix::Bool=false)
    
    matfac = GPUMatFac.matfac_from_hdf(hdf_file, string(path, "/matfac"))
    
    feature_genes = hdf_file[string(path, "/feature_genes")][:]
    feature_assays = hdf_file[string(path, "/feature_assays")][:]

    sample_ids = hdf_file[string(path, "/sample_ids")][:]
    sample_groups = hdf_file[string(path, "/sample_groups")][:]
    
    if load_omic_matrix
        if in("omic_matrix", keys(hdf_file["path"]))
            omic_matrix = hdf_file[string(path, "/omic_matrix")][:,:]
        else
            @warn "No omic_matrix saved in HDF file"
        end
    else
        omic_matrix = nothing
    end

    return MultiomicModel(matfac, feature_genes, feature_assays,
                          sample_ids, sample_groups, omic_matrix)
end

###########################################
# Weighted edge list digraph representation
###########################################

function edgelist_to_hdf(hdf_file, path::String, edgelist::Vector)
    u_vec = [edge[1] for edge in edgelist]
    v_vec = [edge[2] for edge in edgelist]
    w_vec = [edge[3] for edge in edgelist]
    write(hdf_file, string(path, "/u"), u_vec)
    write(hdf_file, string(path, "/v"), v_vec)
    write(hdf_file, string(path, "/w"), w_vec)
    return
end

function edgelist_from_hdf(hdf_file, path::String)
    u_vec = hdf_file[string(path, "/u")][:]
    v_vec = hdf_file[string(path, "/v")][:]
    w_vec = hdf_file[string(path, "/w")][:]
    return [[u, v_vec[i], w_vec[i]] for (i,u) in enumerate(u_vec)]
end

#######################################
# Dictionary
#######################################

function to_hdf(hdf_file, path::String, d::Dict)
    pp = collect(d)
    k_vec = [first(p) for p in pp]
    v_vec = [last(p) for p in pp]
    write(hdf_file, string(path,"/keys"), k_vec)
    write(hdf_file, string(path,"/values"), v_vec)
    return
end

function dictionary_from_hdf(hdf_file, path::String)
    k_vec = hdf_file[string(path,"/keys")][:]
    v_vec = hdf_file[string(path,"/values")][:]
    return Dict(zip(k_vec,v_vec))
end



