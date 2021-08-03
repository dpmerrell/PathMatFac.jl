

using HDF5

import GPUMatFac: to_hdf, model_from_hdf

export save_hdf, load_hdf


function save_hdf(file_path::String, model::MultiomicModel)

end

function load_hdf(file_path::String)

end



######################################
# MultiomicModel
######################################

function to_hdf(hdf_file, path::String, model::MultiomicModel; save_omic_matrix::Bool=false)

    to_hdf(hdf_file, string(path, "/matfac"), model.matfac)
    
    to_hdf(hdf_file, string(path, "/featurenames"), model.featurenames)
    for (i, pwy) in enumerate(model.pwy_graphs)
        edgelist_to_hdf(hdf_file, string(path, "/pwy_graphs/", i), pwy)
    end
    to_hdf(hdf_file, string(path, "/aug_feature_to_idx"), model.aug_feature_to_idx)

    to_hdf(hdf_file, string(path, "/sample_ids"), model.sample_ids)
    edgelist_to_hdf(hdf_file, string(path, "/sample_graph"), model.sample_graph)
    to_hdf(hdf_file, string(path, "/aug_sample_to_idx"), model.aug_sample_to_idx)

    if save_omic_matrix
        write(hdf_file, string(path, "/omic_matrix"), model.omic_matrix)
    end

    return
end


function multiomicmodel_from_hdf(hdf_file)
    matfac = 
end

###########################################
# Weighted edge list digraph representation
###########################################

function edgelist_to_hdf(hdf_file, path::String, edgelist::Vector{Vector})
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
    return [[u, v_vec[i], w_vec[i]] for (u,i) in enumerate(u_vec)]
end

#######################################
# Dictionary
#######################################

function to_hdf(hdf_file, path::String, d::Dictionary)
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



