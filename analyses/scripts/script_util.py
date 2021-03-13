import numpy as np
import h5py

def load_hdf(dataset_hdf, key, dtype=float):
    
    with h5py.File(dataset_hdf, "r") as f:
        dataset = f[key][:]
    
    return dataset.astype(dtype) 


def load_features(dataset_hdf):
    return load_hdf(dataset_hdf, "index", dtype=str)

def load_instances(dataset_hdf):
    return load_hdf(dataset_hdf, "columns", dtype=str)    

def load_data(dataset_hdf):
    return load_hdf(dataset_hdf, "data")


def load_true(truth_hdf):
    true_data = load_data(truth_hdf)
    true_instances = load_instances(truth_hdf)
    true_features = load_features(truth_hdf)

    print("TRUE DATA:", true_data.shape)

    return true_data, true_instances, true_features


def load_factor_hdf(factor_hdf, feat_key="index", inst_key="columns"):

    feat_factor = load_hdf(factor_hdf, "feature_factor")
    inst_factor = load_hdf(factor_hdf, "instance_factor")
    features = load_hdf(factor_hdf, feat_key, dtype=str)
    instances = load_hdf(factor_hdf, inst_key, dtype=str) 
    pathways = load_hdf(factor_hdf, "pathways", dtype=str)

    pathways = np.array([pwy.split("/")[-1] for pwy in pathways])

    print("SHAPES:")
    print("\tfeat_factor:", feat_factor.shape)
    print("\tinst_factor:", inst_factor.shape)
    print("\tfeatures:", features.shape)
    print("\tinstances:", instances.shape)
    print("\tpathways:", pathways.shape)

    return feat_factor, inst_factor, features, instances, pathways



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



