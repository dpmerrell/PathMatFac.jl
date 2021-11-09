import numpy as np
import h5py


NICE_NAMES = {"gender": "Sex",
              "hpv_status": "HPV",
              "tobacco_smoking_history": "Smoking",
              "age_at_initial_pathologic_diagnosis": "Age",
              "race": "Race"
              }

"""
Convert a pathway into a list of gene IDs.
"""
def pwy_to_geneset(pwy):

    gene_set = set([])

    for edge in pwy:
        u = edge[0]
        u_type = edge[1][0]
        if u_type == "p":
            gene_set.add(u)
        
        v = edge[2]
        v_type = edge[1][1]
        if v_type == "p":
            gene_set.add(v)

    return list(gene_set)


def load_hdf(dataset_hdf, key, dtype=float):
    
    with h5py.File(dataset_hdf, "r") as f:
        dataset = f[key][:]
    
    return dataset.astype(dtype) 


def load_features(dataset_hdf):
    return load_hdf(dataset_hdf, "features", dtype=str)

def load_instances(dataset_hdf):
    return load_hdf(dataset_hdf, "instances", dtype=str)    

def load_data(dataset_hdf):
    return load_hdf(dataset_hdf, "data")


def load_true(truth_hdf):
    true_data = load_data(truth_hdf)
    true_instances = load_instances(truth_hdf)
    true_features = load_features(truth_hdf)

    print("TRUE DATA:", true_data.shape)

    return true_data, true_instances, true_features


def load_embedding(model_hdf):

    with h5py.File(model_hdf, "r") as f:
        X = f["matfac"]["X"][:,:]

    return X


def load_sample_info(model_hdf):

    with h5py.File(model_hdf, "r") as f:
        original_samples = f["original_samples"][:].astype(str)
        augmented_samples = f["augmented_samples"][:].astype(str)
        
        original_groups = f["original_groups"][:].astype(str)

    sample_to_idx = {samp: idx for (idx, samp) in enumerate(augmented_samples)}

    return original_samples, original_groups, augmented_samples, sample_to_idx


def load_feature_info(model_hdf):

    with h5py.File(model_hdf, "r") as f:
        original_genes = f["original_genes"][:].astype(str)
        augmented_genes = f["augmented_genes"][:].astype(str)
        
        original_assays = f["original_assays"][:].astype(str)
        augmented_assays = f["augmented_assays"][:].astype(str)

    feat_to_idx = {pair: idx for (idx, pair) in enumerate(zip(augmented_genes, augmented_assays))}

    return original_genes, original_assays, augmented_genes, augmented_assays, feat_to_idx


def load_instance_offset(model_hdf):
    with h5py.File(model_hdf, "r") as f:
        offset = f["matfac"]["instance_offset"][:]
    return offset

def load_feature_offset(model_hdf):
    with h5py.File(model_hdf, "r") as f:
        offset = f["matfac"]["feature_offset"][:]
    return offset

def value_to_idx(ls):
    return {k: idx for idx, k in enumerate(ls)}


def keymatch(l_keys, r_keys):

    rkey_to_idx = value_to_idx(r_keys) 

    l_idx = []
    r_idx = []

    for i, lk in enumerate(l_keys):
        if lk in rkey_to_idx.keys():
            l_idx.append(i)
            r_idx.append(rkey_to_idx[lk])

    return l_idx, r_idx



def feature_to_loss(feature_id):

    if feature_id.endswith("mutation"):
        return "logistic"
    else:
        return "linear"


def omic_feature_losses(feature_ids):
    return [feature_to_loss(feat) for feat in feature_ids]


def parse_arg_str(arg_str):

    kv_strs = arg_str.split("_")
    kvs = [kv.split(":") for kv in kv_strs]
    keys = [kv[0] for kv in kvs]
    val_lists = [kv[1].split(",") for kv in kvs]

    return dict(zip(keys, val_lists))
    


