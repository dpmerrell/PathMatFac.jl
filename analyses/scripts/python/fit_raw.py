

import script_util as su
import numpy as np
import argparse
import h5py
import sys


def load_data(data_hdf, modalities):

    X = None
    modality_set = set(modalities)

    with h5py.File(data_hdf, "r") as f:

        data = f["omic_data"]["data"][:,:].transpose()
        assays = f["omic_data"]["feature_assays"][:].astype(str)

        relevant_cols = np.vectorize(lambda x: x in modality_set)(assays)
        X = data[:,relevant_cols]

        sample_ids = f["omic_data"]["instances"][:].astype(str)
        sample_groups = f["omic_data"]["instance_groups"][:].astype(str)

        feature_genes = f["omic_data"]["feature_genes"][:].astype(str)
        feature_genes = feature_genes[relevant_cols]
        assays = assays[relevant_cols]

        target = f["target"][...].astype(str)

    return X, sample_ids, sample_groups, assays, feature_genes, target


def nan_filter(X, count_axis=0, min_count=20):
    finite_X = np.isfinite(X)
    keep_idx = np.where(np.sum(finite_X, axis=count_axis) > min_count)[0]
    return keep_idx


def variance_filter(X, filter_frac):
    col_vars = np.nanvar(X, axis=0)
    qtile = np.nanquantile(col_vars, 1 - filter_frac)
    keep_idx = np.where(col_vars >= qtile)[0]
    return keep_idx


def filter_assay_cols(X, feature_assays, min_finite_count=20, var_filter_frac=0.05):

    unq_a = np.unique(feature_assays)
    all_kept_idx = []

    for ua in unq_a:
        # For this assay...
        rel_idx = np.where(feature_assays == ua)[0]
        rel_X = X[:,rel_idx]

        # Filter by NaNs
        nonnan_idx = nan_filter(rel_X, min_count=min_finite_count)
        rel_idx = rel_idx[nonnan_idx]
        rel_X = X[:,rel_idx]

        # Filter by variance
        highvar_idx = variance_filter(rel_X, var_filter_frac)
        rel_idx = rel_idx[highvar_idx]

        # keep the result
        all_kept_idx.append(rel_idx)

    conc_kept_idx = np.concatenate(all_kept_idx)
    conc_kept_idx = np.sort(conc_kept_idx)

    return conc_kept_idx

def median_impute(X):
    col_meds = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        X[np.isnan(X[:,j]),j] = col_meds[j]

    return col_meds, X  


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("data_hdf", help="HDF5 containing tabular data")
    parser.add_argument("fitted_hdf")
    parser.add_argument("transformed_data_hdf")
    parser.add_argument("--omic_types", help="Use the specified omic assays.", default="mutation:methylation:mrnaseq:cna")
    parser.add_argument("--var_filter", help="Discard the features with least variance, *keeping* this fraction of the features.", type=float, default=0.5)

    args = parser.parse_args()

    data_hdf = args.data_hdf
    fitted_hdf = args.fitted_hdf
    trans_hdf = args.transformed_data_hdf
    v_frac = args.var_filter
    omic_types = args.omic_types.split(":")

    # Load data
    Z, sample_ids, sample_groups, feature_assays, feature_genes, target = load_data(data_hdf, omic_types)

    # Filter columns by missingness and variance
    keep_idx = filter_assay_cols(Z, feature_assays, 
                                 var_filter_frac=v_frac)
    print("Column filtering: ", Z.shape[1], "-->", len(keep_idx))
    Z = Z[:,keep_idx]
    col_medians, Z = median_impute(Z)
    feature_assays = feature_assays[keep_idx]
    feature_genes = feature_genes[keep_idx]

    print("TRAINING DATA:")
    print(Z.shape)

    # Output the transformed data and the 
    # fitted principal components and standardization parameters
    with h5py.File(trans_hdf, "w", driver="core") as f:
        su.write_hdf(f, "X", Z.transpose())
        su.write_hdf(f, "instances", sample_ids, is_string=True) 
        su.write_hdf(f, "instance_groups", sample_groups, is_string=True)
        su.write_hdf(f, "target", target, is_string=True) 
    
    with h5py.File(fitted_hdf, "w", driver="core") as f:
        su.write_hdf(f, "feature_medians", col_medians)
        su.write_hdf(f, "feature_assays", feature_assays, is_string=True) 
        su.write_hdf(f, "feature_genes", feature_genes, is_string=True) 
        

