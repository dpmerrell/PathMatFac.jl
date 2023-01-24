
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

        target = f["target"][...].astype(str)

    return X, sample_ids, sample_groups, assays, feature_genes, target



def remove_missing(X, out_dim):

    finite_X = np.isfinite(X)
    row_key = (np.sum(finite_X, axis=1) > out_dim)
    col_key = (np.sum(finite_X, axis=0) > out_dim)

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
                    method='nipals',
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
    parser.add_argument("fitted_hdf")
    parser.add_argument("transformed_train_hdf")
    parser.add_argument("--output_dim", help="Use the top-`output_dim` principal components.", type=int)
    parser.add_argument("--variance_filter", help="Discard the features with least variance, *keeping* this fraction of the features.", type=float, default=0.5)
    args = parser.parse_args()

    data_hdf = args.data_hdf
    fitted_hdf = args.fitted_hdf
    trans_hdf = args.transformed_train_hdf
    output_dim = args.output_dim
    v_frac = args.variance_filter

    # Load data
    Z, sample_ids, sample_groups, feature_assays, feature_genes, target = load_data(data_hdf)

    # Remove empty rows and columns
    Z_nomissing, row_key, col_key = remove_missing(Z, output_dim)
    sample_ids = sample_ids[row_key]
    sample_groups = sample_groups[row_key]
    feature_assays = feature_assays[col_key]
    feature_genes = feature_genes[col_key]

    # Remove the columns with least variance
    Z_filtered, col_key = variance_filter(Z_nomissing, v_frac)
    feature_assays = feature_assays[col_key]
    feature_genes = feature_genes[col_key]
    
    # Standardize the remaining columns
    Z_std, mu, sigma = standardize_columns(Z_filtered)
    
    # Perform PCA
    X, pcs = transform_data(Z_std, n_components=output_dim)

    # Output the transformed data and the 
    # fitted principal components and standardization parameters
    with h5py.File(trans_hdf, "w") as f:
        su.write_hdf(f, "X", X.transpose())
        su.write_hdf(f, "instances", sample_ids, is_string=True) 
        su.write_hdf(f, "instance_groups", sample_groups, is_string=True)
        su.write_hdf(f, "target", target, is_string=True) 
    
    with h5py.File(fitted_hdf, "w") as f:
        su.write_hdf(f, "Y", pcs)
        su.write_hdf(f, "mu", mu)
        su.write_hdf(f, "sigma", sigma)
        su.write_hdf(f, "feature_assays", feature_assays, is_string=True) 
        su.write_hdf(f, "feature_genes", feature_genes, is_string=True) 
        

