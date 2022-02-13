import h5py
import numpy as np
import argparse
import sys


def parse_feature_names(feature_names):

    feature_genes = np.vectorize(lambda x: x.split("_")[0])(feature_names)
    feature_assays = np.vectorize(lambda x: x.split("_")[-1])(feature_names)

    return feature_genes, feature_assays


def load_data(input_hdf):

    with h5py.File(input_hdf, "r") as f:

        omic_matrix = f["omic_data"]["data"][:,:]
        sample_ids = f["omic_data"]["instances"][:].astype(str)
        sample_groups = f["omic_data"]["cancer_types"][:].astype(str)
        feature_names = f["omic_data"]["features"][:].astype(str)

    feature_genes, feature_assays = parse_feature_names(feature_names)


    return omic_matrix, sample_ids, sample_groups, feature_genes, feature_assays


def load_barcodes(input_hdf):

    barcodes = {}
    with h5py.File(input_hdf, "r") as f:
        for k in ("data", "features", "instances"):
            barcodes[k] = f["barcodes"][k][...]

    return barcodes


def hdf_write_string_arr(f_out, path, array):
    dataset = f_out.create_dataset(path, shape=array.shape,
                                   dtype=h5py.string_dtype("utf-8"))
    dataset[...] = array
    return


def output_to_hdf(output_hdf, omic_matrix, sample_ids, sample_groups, 
                  feature_genes, feature_assays, barcodes):

    with h5py.File(output_hdf, "w") as f:
        hdf_write_string_arr(f, "omic_data/instances", sample_ids)
        hdf_write_string_arr(f, "omic_data/instance_groups", sample_groups)
        hdf_write_string_arr(f, "omic_data/feature_genes", feature_genes)
        hdf_write_string_arr(f, "omic_data/feature_assays", feature_assays)
       
        dataset = f.create_dataset("omic_data/data", shape=omic_matrix.shape,
                                           dtype=float)
        dataset[:,:] = omic_matrix

        for k in barcodes.keys():
            hdf_write_string_arr(f, f"barcodes/{k}", barcodes[k])


    return


def inv_logistic(a):
    return np.log(a / (1.0 - a))


def cna_threshold(a, l=-0.5, u=0.5):
    nan_idx = np.isnan(a)
    l_idx = (a <= l)
    u_idx = (a > u)
    mid_idx = np.logical_not( l_idx | u_idx | nan_idx )

    a[l_idx] = 0.0
    a[u_idx] = 1.0
    a[mid_idx] = 0.5

    return a


def mut_threshold(a, u=0.0):
    u_idx = (a > u)
    a[u_idx] = 1.0
    return a


def standardize_by_gp(a, groups):

    unq_gps = np.unique(groups)
    for gp in unq_gps:
        gp_idx = (groups == gp)
        mu = np.nanmean(a[:,gp_idx], axis=1)[:,np.newaxis]
        std = np.nanstd(a[:,gp_idx], axis=1)[:,np.newaxis]
        a[:,gp_idx] = (a[:,gp_idx] - mu)/std

    return a


def preprocess_features(omic_matrix, feature_assays, sample_groups, standardized_assays):

    assays = ["methylation", "cna", "mutation", "mrnaseq", "rppa"]
    assay_rows = {assay: (feature_assays == assay) for assay in assays} 

    print("ASSAY ROWS:", assay_rows)

    omic_matrix[assay_rows["methylation"],:] = inv_logistic(omic_matrix[assay_rows["methylation"],:])
    omic_matrix[assay_rows["cna"],:] = cna_threshold(omic_matrix[assay_rows["cna"],:])
    omic_matrix[assay_rows["mutation"],:] = mut_threshold(omic_matrix[assay_rows["mutation"],:])

    for std_assay in standardized_assays:
        omic_matrix[assay_rows[std_assay],:] = standardize_by_gp(omic_matrix[assay_rows[std_assay],:], sample_groups)

    return omic_matrix


def parse_opts(opt_ls):
    
    opt_kv = [opt.split("=") for opt in opt_ls]
    opt_k = [kv[0] for kv in opt_kv]
    opt_v = [kv[1].split(":") for kv in opt_kv]
    for i, v in enumerate(opt_v):
        if v == [""]:
            opt_v[i] = []

    return dict(zip(opt_k, opt_v))


if __name__=="__main__":

    args = sys.argv
    
    input_hdf = args[1]
    output_hdf = args[2]

    opt_dict = parse_opts(args[3:])

    heldout_ctypes = opt_dict["heldout_ctypes"]
    standardized_assays = opt_dict["std_assays"]

    print("STD_ASSAYS:", standardized_assays)

    omic_matrix,\
    sample_ids, sample_groups, \
    feature_genes, feature_assays = load_data(input_hdf)

    barcodes = load_barcodes(input_hdf)

    good_idx = np.ones(len(sample_groups),dtype=bool)
    for ct in heldout_ctypes:
        good_idx = (good_idx & np.logical_not(sample_groups == ct))

    omic_matrix = omic_matrix[:,good_idx]
    sample_groups = sample_groups[good_idx]
    sample_ids = sample_ids[good_idx]

    barcodes["instances"] = barcodes["instances"][good_idx]
    barcodes["data"] = barcodes["data"][:,good_idx]

    print(omic_matrix.shape)
    print(sample_groups.shape)
    print(feature_assays.shape)

    prepped_omics = preprocess_features(omic_matrix, feature_assays, sample_groups, standardized_assays)
    
    output_to_hdf(output_hdf, prepped_omics, 
                              sample_ids, sample_groups,
                              feature_genes, feature_assays,
                              barcodes)

    

