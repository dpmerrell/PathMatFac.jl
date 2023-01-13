
from statsmodels.multivariate.pca import PCA
import script_util as su
import numpy as np
import argparse
import h5py
import sys

def load_data(data_hdf, modality="mrnaseq"):

    X = None

    with h5py.File(data_hdf, "r") as f:

        data = f["omic_data"]["data"][:,:].transpose()
        assays = f["omic_data"]["feature_assays"][:].astype(str)

        relevant_cols = (assays == modality)
        X = data[:,relevant_cols]

        sample_ids = f["omic_data"]["instances"][:].astype(str)
        sample_groups = f["omic_data"]["instance_groups"][:].astype(str)

        feature_genes = f["omic_data"]["feature_genes"][:].astype(str)
        feature_genes = feature_genes[relevant_cols]
        assays = assays[relevant_cols]

    return X, sample_ids, sample_groups, assays, feature_genes



def remove_missing(X):

    row_key = np.zeros(X.shape[0], dtype=bool)
    for i in range(X.shape[0]):
        if np.any(np.isfinite(X[i,:])):
            row_key[i] = True

    col_key = np.zeros(X.shape[1], dtype=bool)
    for j in range(X.shape[1]):
        if np.any(np.isfinite(X[:,j])):
            col_key[j] = True

    X_nomissing = X[row_key,:]
    X_nomissing = X_nomissing[:,col_key]

    return X_nomissing, row_key, col_key


def variance_filter(X, filter_frac=0.5):

    col_vars = np.nanvar(X, axis=0)
    qtile = np.nanquantile(col_vars, 1 - filter_frac)
    keep_idx = (col_vars >= qtile)

    return X[:,keep_idx], keep_idx
   

def standardize_columns(X):

    col_std = np.nanstd(X, axis=0)
    col_means = np.nanmean(X, axis=0)

    X = (X - col_means.transpose()) / col_std.transpose()

    return X, col_means, col_std 


def find_knee(rsquare):

    # Find maximum of discrete 2nd derivative
    d1 = rsquare[1:] - rsquare[:-1]
    d2 = d1[1:] - d1[:-1]
    
    # Off-by-one because of finite difference
    max_idx = np.argmax(d2) + 1
    return max_idx


def transform_data(X, n_components=None):

    result = PCA(X, standardize=False,
                    demean=False,
                    normalize=False,
                    ncomp=n_components, 
                    missing="fill-em",
                    max_em_iter=500)

    X_trans = result.factors
    rsquare = result.rsquare
    pcs = result.loadings

    if n_components is None:
        knee_idx = find_knee(rsquare)
        X_trans = X_trans[:,:knee_idx]

    return X_trans, pcs


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("data_hdf", help="HDF5 containing tabular data")
    parser.add_argument("transformed_hdf")
    parser.add_argument("fitted_pca_hdf")
    parser.add_argument("--output_dim", help="Use the top-`output_dim` principal components.", type=int)
    parser.add_argument("--variance_filter", help="Discard the features with least variance, *keeping* this fraction of the features.", type=float, default=0.5)
    args = parser.parse_args()

    data_hdf = args.data_hdf
    transformed_hdf = args.transformed_hdf
    pcs_hdf = args.fitted_pca_hdf
    output_dim = args.output_dim
    v_frac = args.variance_filter

    # Load data
    X, sample_ids, sample_groups, feature_assays, feature_genes = load_data(data_hdf)

    print("X SHAPE:")
    print(X.shape)

    # Remove empty rows and columns
    X_nomissing, row_key, col_key = remove_missing(X)
    sample_ids = sample_ids[row_key]
    sample_groups = sample_groups[row_key]
    feature_assays = feature_assays[col_key]
    feature_genes = feature_genes[col_key]

    print("X_NOMISSING SHAPE:")
    print(X_nomissing.shape)

    # Remove the columns with least variance
    X_filtered, col_key = variance_filter(X_nomissing, v_frac)
    feature_assays = feature_assays[col_key]
    feature_genes = feature_genes[col_key]
    
    print("X_FILTERED SHAPE:")
    print(X_filtered.shape)

    # Standardize the remaining columns
    X_std, mu, sigma = standardize_columns(X_filtered)
    print("X_STD SHAPE:")
    print(X_std.shape)

    # Perform PCA
    X_transformed, pcs = transform_data(X_std, n_components=output_dim)

    # Output the transformed data 
    with h5py.File(transformed_hdf, "w") as f:
        su.write_hdf(f, "X", X_transformed)
        su.write_hdf(f, "instances", sample_ids, is_string=True) 
        su.write_hdf(f, "instance_groups", sample_groups, is_string=True) 

    # Output the fitted principal components and standardization parameters
    with h5py.File(pcs_hdf, "w") as f:
        su.write_hdf(f, "Y", pcs)
        su.write_hdf(f, "mu", mu)
        su.write_hdf(f, "sigma", sigma)
        su.write_hdf(f, "feature_assays", feature_assays, is_string=True) 
        su.write_hdf(f, "feature_genes", feature_genes, is_string=True) 
        

