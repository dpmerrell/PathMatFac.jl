
import script_util as su
import numpy as np
import argparse
import h5py
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


def inv_logistic(a, shrinkage=0.99):
    x = 0.5 + (a - 0.5)*shrinkage
    return np.log(x / (1.0 - x))


def cna_threshold_quantile(a, lq=0.333, uq=0.667):

    m = a.shape[0]
    l = np.reshape(np.nanquantile(a, lq, axis=1), (m,1))
    u = np.reshape(np.nanquantile(a, uq, axis=1), (m,1))
    nan_idx = np.isnan(a)
    l_idx = (a <= l)
    u_idx = (a > u)
    mid_idx = np.logical_not( l_idx | u_idx | nan_idx )

    a[l_idx] = 1.0
    a[u_idx] = 3.0
    a[mid_idx] = 2.0

    return a

def cna_threshold(a, l=-0.5, u=0.5):

    m = a.shape[0]
    nan_idx = np.isnan(a)
    l_idx = (a <= l)
    u_idx = (a > u)
    mid_idx = np.logical_not( l_idx | u_idx | nan_idx )

    a[l_idx] = 1.0
    a[u_idx] = 3.0
    a[mid_idx] = 2.0

    return a

def mut_threshold(a, u=0.0):
    u_idx = (a > u)
    a[u_idx] = 1.0
    return a


def preprocess_features(omic_matrix, feature_assays, sample_groups):

    assays = ["methylation", "cna", "mutation", "mrnaseq", "rppa"]
    assay_rows = {assay: (feature_assays == assay) for assay in assays} 

    print("ASSAY ROWS:", assay_rows)

    omic_matrix[assay_rows["methylation"],:] = inv_logistic(omic_matrix[assay_rows["methylation"],:])
    omic_matrix[assay_rows["cna"],:] = cna_threshold(omic_matrix[assay_rows["cna"],:])
    omic_matrix[assay_rows["mutation"],:] = mut_threshold(omic_matrix[assay_rows["mutation"],:])

    return omic_matrix


def parse_opts(opt_ls, defaults):
   
    print(opt_ls) 
    opt_kv = [opt.split("=") for opt in opt_ls if "=" in opt]
    print(opt_kv)
    opt_k = [kv[0] for kv in opt_kv]
    opt_v = [kv[1].split(":") for kv in opt_kv]
    for i, v in enumerate(opt_v):
        if v == [""]:
            opt_v[i] = []

    for k,v in zip(opt_k, opt_v):
        defaults[k] = v

    return defaults 


if __name__=="__main__":

    args = sys.argv
    
    input_hdf = args[1]
    output_hdf = args[2]


    defaults = {"heldout_ctypes": [],
                "kept_ctypes": su.ALL_CTYPES,
                }

    opt_dict = parse_opts(args[3:], defaults)

    heldout_ctypes = opt_dict["heldout_ctypes"]
    kept_ctypes = opt_dict["kept_ctypes"]

    omic_matrix,\
    sample_ids, sample_groups, \
    feature_genes, feature_assays = load_data(input_hdf)

    barcodes = load_barcodes(input_hdf)

    good_idx = np.ones(len(sample_groups),dtype=bool)
    for ct in heldout_ctypes:
        good_idx = (good_idx & np.logical_not(sample_groups == ct))

    kept_idx = np.zeros(len(sample_groups),dtype=bool)
    for ct in kept_ctypes:
        kept_idx = (kept_idx | (sample_groups == ct))
    good_idx = (good_idx & kept_idx)

    omic_matrix = omic_matrix[:,good_idx]
    sample_groups = sample_groups[good_idx]
    sample_ids = sample_ids[good_idx]

    barcodes["instances"] = barcodes["instances"][good_idx]
    barcodes["data"] = barcodes["data"][:,good_idx]

    print(omic_matrix.shape)
    print(sample_groups.shape)
    print(feature_assays.shape)

    prepped_omics = preprocess_features(omic_matrix, feature_assays, sample_groups)
    
    output_to_hdf(output_hdf, prepped_omics, 
                              sample_ids, sample_groups,
                              feature_genes, feature_assays,
                              barcodes)

    

