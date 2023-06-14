
from collections import defaultdict
from matplotlib import pyplot as plt
import matplotlib as mpl
from os import path
import numpy as np
import pandas as pd
import h5py


NICE_NAMES = {"gender": "Sex",
              "survival": "Survival",
              "ctype": "Cancer type",
              "pathologic_stage": "Pathologic stage",
              "hpv_status": "HPV status",
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
              "X_pwy_spearman_corr": "Pathway Activation Spearman",
              "X_spansim": "$X$ span sim.",
              "Y_spansim": "$Y$ span sim.",
              "A_aucpr": "$A$ AUCPR",
              "roc_auc": "AUCROC",
              "mse": "MSE",
              "accuracy": "Accuracy",
              "concordance": "Concordance",
              "matfac": "PathMatFac",
              "matfac_batch": "P.M.F. (batch)",
              "matfac_nobatch": "P.M.F. (no batch)",
              "mofa": "MOFA+",
              "plier": "PLIER",
              "pca": "PCA",
              "raw": "Raw data",
              "paradigm": "PARADIGM",
              "gsva": "GSVA",
              "mutation": "Mutation",
              "cna": "CNA",
              "methylation": "Methylation",
              "rppa": "RPPA",
              "mrnaseq": "RNA-seq",
              "EB": "Emp. Bayes",
              "EM": "Exp. Max.",
              "LSQ": "Least Sq.",
              "theta_r2": "$R^2$",
              "logdelta_spearman": "Spearman"
              }


ALL_CTYPES = ["ACC", "CESC", "HNSC", 
              "KIRC", "LGG", "LUSC", "PAAD", "READ", "STAD", 
              "THCA", "UCS", "BLCA", "CHOL", "DLBC", "GBM", 
              "KICH", "KIRP", "LIHC", "MESO", "PCPG", "SARC", 
              "THYM", "UVM", "BRCA", "COAD", "ESCA", 
              "LAML", "LUAD", "OV", "PRAD", 
              "SKCM", "TGCT", "UCEC"]

ALL_COLORS = ["red","blue","black","orange","yellow","grey","silver"]

ALL_TARGETS = ["survival","ctype","pathologic_stage","hpv_status"]

ALL_METHODS = ["raw", "matfac", "matfac_batch", "matfac_nobatch", "mofa", "plier", "pca", "paradigm", "gsva"]

ALL_SCORES = {"survival": "concordance",
              "ctype": "accuracy",
              "pathologic_stage": "mse",
              "hpv_status": "roc_auc"}

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
    

def parse_path_kvs(filepath, kv_sep="__"):

    result = {}
    dir_strs = filepath.split(path.sep)
    # Exclude file extension
    dir_strs[-1] = ".".join(dir_strs[-1].split(".")[:-1]) 
    for dir_str in dir_strs:
        for kv_str in dir_str.split(kv_sep):
            k_v = kv_str.split("=")
            if len(k_v) == 2:
                k = k_v[0]
                v = k_v[1]
                if k in result.keys():
                    k += "_+" 

                result[k] = parse_value(v)

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


def linear_transform(Z, Y, lr=1.0, rel_tol=1e-6, max_iter=1000):

    K, N = Y.shape
    M = Z.shape[0]

    X = np.zeros((K, M))
    grad_X = np.zeros((K,M))
    grad_ssq = np.zeros((K,M)) + 1e-8

    nan_idx = np.logical_not(np.isfinite(Z))
    lss = np.inf
    i = 0

    # Apply Adagrad updates until convergence...
    while i < max_iter:
        new_lss = 0.0
            
        # Compute the gradient of squared loss w.r.t. X
        delta = np.dot(X.transpose(), Y) - Z
        delta[nan_idx] = 0.0
        grad_X = np.dot(Y, delta.transpose())
  
        # Update the sum of squared gradients
        grad_ssq += grad_X*grad_X

        # Apply the update
        X -= lr*(grad_X / np.sqrt(grad_ssq))
      
        # Compute the loss 
        np.square(delta, out=delta) 
        new_lss += np.sum(delta)

        # Check termination criterion
        if abs((lss - new_lss)/new_lss) < rel_tol:
            print("Loss decrease < rel tol ({}). Terminating".format(rel_tol))
            break

        # Update loop variables
        lss = new_lss
        if i % 100 == 0: 
            print("Iteration: {}; Loss: {:.2f}".format(i, lss))
        i += 1

    if i >= max_iter:
        print("Reached max iter ({}). Terminating".format(max_iter))

    return X


def sort_by_order(ls, ordered_vocab):
    ordering_dict = {x:i for i,x in enumerate(ordered_vocab)}
    srt_idx = sorted([ordering_dict[x] for x in ls])
    return [ordered_vocab[i] for i in srt_idx]


def get_method_target(result_json):
    path_kvs = parse_path_kvs(result_json)
    method = path_kvs["method"]
    target = path_kvs["target"]
    return method, target


def get_methods_targets(result_jsons):
    """
    Given a list of JSON filepaths, store them in 
    a nested dictionary indexed by "method" and "target".
    """
    result = defaultdict(lambda : defaultdict(list)) 
    for rjs in result_jsons:
        method, target = get_method_target(rjs)
        result[method][target].append(rjs)

    return result
    

def dict_to_grid(d, row_order=ALL_METHODS, col_order=ALL_TARGETS):

    rownames = list(d.keys())
    colnames = set()
    for rname in rownames:
        colnames |= d[rname].keys()
   
    rownames = sort_by_order(rownames, row_order)
    colnames = sort_by_order(list(colnames), col_order)
    
    mat = [[sorted(d[rname][cname]) for cname in colnames] for rname in rownames]

    return mat, rownames, colnames


def make_subplot_grid(plt_func, grid, rownames, colnames, figsize=None):
    """
    Construct a set of subplots populated with data from `grid`.
        * `grid`: a list of lists of data indexed by (row, column, data).
        * `rownames`, `colnames`: string labels for the grid rows and columns
        * `plt_func`: a function that receives an `axes` object and an entry from `grid`;
                      and mutates the `axes` object (i.e., calls pyplot functions 
                      on it, using the data contained in the list.)
    """

    n_rows = len(rownames)
    n_cols = len(colnames)

    if figsize is None:
        figsize=(2.0*n_cols, 2.0*n_rows)

    fig, axarr = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_rows > 1:
        for i, rowname in enumerate(rownames):
            for j, colname in enumerate(colnames):
                ax = axarr[i][j]
                plt_func(ax, grid[i][j])
    else:
        for j, colname in enumerate(colnames):
            ax = axarr[j]
            plt_func(ax, grid[0][j])
           
    return fig, axarr



