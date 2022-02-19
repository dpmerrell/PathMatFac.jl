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
        dataset = f[key][...]
    
    return dataset.astype(dtype) 


def load_embedding(model_hdf):

    with h5py.File(model_hdf, "r") as f:
        internal_X = f["matfac"]["X"][:,:]
        sample_idx = f["internal_sample_idx"][:]
        X = internal_X[sample_idx,:]

    return X


def load_sample_ids(model_hdf):
    return load_hdf(model_hdf, "sample_ids", dtype=str)


def load_sample_groups(model_hdf):
    return load_hdf(model_hdf, "sample_conditions", dtype=str)

def load_pathway_names(model_hdf):
    return load_hdf(model_hdf, "pathway_names", dtype=str)

def load_features(model_hdf):
    feature_genes = load_hdf(model_hdf, "feature_genes", dtype=str)
    feature_assays = load_hdf(model_hdf, "feature_assays", dtype=str)
    f_l = np.array(list("{}_{}".format(g,a) for (g,a) in zip(feature_genes, feature_assays)))

    feature_idx = load_hdf(model_hdf, "feature_idx", dtype=int)
    feature_idx -= 1
    return f_l[feature_idx]


def load_feature_factors(model_hdf):
    factors = load_hdf(model_hdf, "matfac/Y")
    feature_idx = load_hdf(model_hdf, "internal_feature_idx", dtype=int)
    feature_idx -= 1 # convert from Julia to Python indexing!
    return factors[feature_idx, :]


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
    


