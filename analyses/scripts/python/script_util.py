
from os import path
import numpy as np
import pandas as pd
import h5py


NICE_NAMES = {"gender": "Sex",
              "hpv_status": "HPV",
              "tobacco_smoking_history": "Smoking",
              "age_at_initial_pathologic_diagnosis": "Age",
              "race": "Race",
              "nadd": "$N_a$",
              "nrem": "$N_r$",
              "kadd": "$K_a$",
              "krem": "$K_r$",
              "snr": "SNR",
              "missing": "Missing Data",
              "l1_fraction": "$L^1$ Fraction",
              "X_pwy_spearman_corr": "Pathway Activation Spearman"
              }

ALL_CTYPES = ["ACC", "CESC", "HNSC", 
              "KIRC", "LGG", "LUSC", "PAAD", "READ", "STAD", 
              "THCA", "UCS", "BLCA", "CHOL", "DLBC", "GBM", 
              "KICH", "KIRP", "LIHC", "MESO", "PCPG", "SARC", 
              "THYM", "UVM", "BRCA", "COAD", "ESCA", 
              "LAML", "LUAD", "OV", "PRAD", 
              "SKCM", "TGCT", "UCEC"]

"""
Convert a pathway into a list of gene IDs.
"""
def pwy_to_geneset(pwy, all_genes):

    gene_set = set([])

    for edge in pwy:
        for entity in edge[:2]:
            if entity in all_genes:
                gene_set.add(entity)

    return list(gene_set)


def load_hdf(dataset_hdf, key, dtype=float):
    
    with h5py.File(dataset_hdf, "r") as f:
        dataset = f[key][...].astype(dtype)
    
    return dataset 


def load_hdf_dict(dataset_hdf, key, keytype=str, dtype=float):

    with h5py.File(dataset_hdf, "r") as f:
        keys = f[key]["keys"][:].astype(keytype)
        values = f[key]["values"][:].astype(dtype)

        my_dict = dict(zip(keys,values))

    return my_dict


def write_hdf(hdf_obj, key, data, is_string=False):
    if is_string:
        hdf_obj.create_dataset(key, shape=data.shape, dtype=h5py.string_dtype())
        hdf_obj[key][...] = data
    else:
        hdf_obj.create_dataset(key, data=data)
        


def ids_to_idx_dict(id_vec):

    _, unq_idx = np.unique(id_vec, return_index=True)
    unq_ids = id_vec[np.sort(unq_idx)]
    idx_dict = {ui:[] for ui in unq_ids}
    for i, name in enumerate(id_vec):
        idx_dict[name].append(i)

    return idx_dict



def load_batch_matrix(model_hdf, bmf_key, values_key, keytype=str, dtype=float):

    feature_batch_ids = load_hdf(model_hdf, f"{bmf_key}/feature_batch_ids", dtype=str)
    _, unq_idx = np.unique(feature_batch_ids, return_index=True)
    feature_batch_ids = feature_batch_ids[np.sort(unq_idx)]

    nfb = len(feature_batch_ids)

    sample_batch_ids = [load_hdf(model_hdf, f"{bmf_key}/sample_batch_ids/{idx+1}", dtype=str) for idx in range(nfb)]

    batch_value_dicts = [load_hdf_dict(model_hdf, f"{values_key}/{idx+1}", keytype=str, dtype=float) for idx in range(nfb)]

    internal_sample_idx = load_hdf(model_hdf, "internal_sample_idx", dtype=int) 

    return feature_batch_ids, sample_batch_ids, batch_value_dicts, internal_sample_idx



def load_embedding(param_hdf):
    return load_hdf(param_hdf, "X") 


def load_sample_ids(param_hdf):
    return load_hdf(param_hdf, "sample_ids", dtype=str)

def load_sample_groups(param_hdf):
    return load_hdf(param_hdf, "sample_conditions", dtype=str)

def load_pathway_names(param_hdf):
    return load_hdf(param_hdf, "pathway_names", dtype=str)

def load_features(param_hdf):
    feature_genes = load_hdf(param_hdf, "data_genes", dtype=str)
    feature_assays = load_hdf(param_hdf, "data_assays", dtype=str)
    f_l = np.array(list("{}_{}".format(g,a) for (g,a) in zip(feature_genes, feature_assays)))

    feature_idx = load_hdf(param_hdf, "used_feature_idx", dtype=int)
    feature_idx -= 1
    return f_l[feature_idx]


def load_feature_factors(param_hdf):
    factors = load_hdf(param_hdf, "Y")
    return factors


def load_col_param(param_hdf, key):
    raw_param = load_hdf(param_hdf, key)
    internal_idx = load_hdf(param_hdf, "used_feature_idx", dtype=int)
    internal_idx -= 1
    return raw_param[internal_idx]

def load_mu(param_hdf):
    return load_col_param(param_hdf, "matfac/mu")

def load_log_sigma(param_hdf):
    return load_col_param(param_hdf, "matfac/log_sigma")

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

def parse_value(v):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v
    

def parse_path_kvs(pth):

    result = {}
    dir_strs = pth.split(path.sep)
    dir_strs[-1] = dir_strs[-1].split(".")[0]
    for dir_str in dir_strs:
        sep = "__"
        for kv_str in dir_str.split(sep):
            k_v = kv_str.split("=")
            if len(k_v) == 2:
                result[k_v[0]] = parse_value(k_v[1])

    return result


def groupby_except(df, except_cols):
    except_set = set(except_cols)
    gp_cols = [col for col in df.columns if col not in except_set]
    return df.groupby(gp_cols)


def aggregate_replicates(df, replicate_cols, op="mean"):
    gp = groupby_except(df, replicate_cols)
    result = None
    if op == "mean":
        result = gp.mean()
    elif op == "var":
        result = gp.var()
    elif op == "median":
        result = gp.median()
    elif op == "count":
        result = gp.count()
    elif op == "std":
        result = gp.std()
    else:
        raise ValueError
    result = pd.DataFrame(result) 
    result.reset_index(inplace=True)
    return result


stage_enc_dict = {"stage i": 0,
                  "stage ii": 1,
                  "stage iii": 2,
                  "stage iv": 3,
                  "stage v": 4} 

def stage_encoder(stage_str):
    if isinstance(stage_str, bytes):
        stage_str = stage_str.decode()
    if not (stage_str[-1] in ("i","v")):
        stage_str = stage_str[:-1]
    return stage_enc_dict[stage_str]


def encode_pathologic_stage(y):
    return np.vectorize(stage_encoder)(y)

