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
def pwy_to_geneset(pwy, all_genes):

    gene_set = set([])

    for edge in pwy:
        for entity in edge[:2]:
            if entity in all_genes:
                gene_set.add(entity)

    return list(gene_set)


def load_hdf(dataset_hdf, key, dtype=float):
    
    with h5py.File(dataset_hdf, "r") as f:
        dataset = f[key][...]
    
    return dataset.astype(dtype) 


def load_hdf_dict(dataset_hdf, key, keytype=str, dtype=float):

    with h5py.File(dataset_hdf, "r") as f:
        keys = f[key]["keys"][:].astype(keytype)
        values = f[key]["values"][:].astype(dtype)

        my_dict = dict(zip(keys,values))

    return my_dict


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




