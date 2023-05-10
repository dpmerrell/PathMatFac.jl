
from statsmodels.multivariate.pca import PCA
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

        #relevant_cols = (assays == modality)
        relevant_cols = np.vectorize(lambda x: x in modality_set)(assays)
        X = data[:,relevant_cols]

        sample_ids = f["omic_data"]["instances"][:].astype(str)
        sample_groups = f["omic_data"]["instance_groups"][:].astype(str)

        feature_genes = f["omic_data"]["feature_genes"][:].astype(str)
        feature_genes = feature_genes[relevant_cols]
        assays = assays[relevant_cols]

        target = f["target"][...].astype(str)

    return X, sample_ids, sample_groups, assays, feature_genes, target



def remove_missing(X, out_dim=20):

    finite_X = np.isfinite(X)
    row_key = (np.sum(finite_X, axis=1) > out_dim)
    col_key = (np.sum(finite_X, axis=0) > out_dim)

    X_nomissing = X[row_key,:]
    X_nomissing = X_nomissing[:,col_key]

    return X_nomissing, row_key, col_key


def assay_variance_filter(X, filter_frac):
    
    col_vars = np.nanvar(X, axis=0)
    qtile = np.nanquantile(col_vars, 1 - filter_frac)
    keep_idx = (col_vars >= qtile)
    return keep_idx


def variance_filter(X, feature_assays, filter_frac):

    unq_a = np.unique(feature_assays)
    all_kept_idx = []

    for ua in unq_a:
        rel_idx = np.where(feature_assays == ua)
        rel_X = X[:,rel_idx]
        kept_idx = assay_variance_filter(rel_X, filter_frac)

    conc_kept_idx = np.concatenate(kept_idx)

    return conc_kept_idx
  

def standardize_columns(X):

    col_std = np.nanstd(X, axis=0)
    col_means = np.nanmean(X, axis=0)

    X = (X - col_means.transpose()) / col_std.transpose()

    return X, col_means, col_std 


def find_elbow(rsquare):

    # Find maximum of discrete 2nd derivative
    d1 = rsquare[1:] - rsquare[:-1]
    d2 = d1[1:] - d1[:-1]
    
    # Off-by-one because of finite difference
    max_idx = np.argmax(d2) + 1
    return max_idx


def transform_data(X, min_components=3, max_components=20):

    result = PCA(X, standardize=False,
                    method='nipals',
                    demean=False,
                    normalize=False,
                    ncomp=max_components, 
                    missing="fill-em",
                    max_em_iter=5)

    X_trans = result.factors
    rsquare = result.rsquare
    pcs = result.loadings

    elbow_idx = find_elbow(rsquare)
    elbow_idx = max(min_components, elbow_idx)

    X_trans = X_trans[:,:elbow_idx]
    pcs = pcs[:,:elbow_idx]

    return X_trans, pcs


# Construct a block-diagonal matrix containing the
# concatenated principal vectors
def concatenate_pcs(assay_Y):

    K_ls = [y.shape[0] for y in assay_Y]
    N_ls = [y.shape[1] for y in assay_Y]

    acc_K = np.cumsum(K_ls)
    acc_N = np.cumsum(N_ls)
    conc_Y = np.zeros((acc_K[-1], acc_N[-1]))

    for i, Y in enumerate(assay_Y):
        prev_k = 0
        prev_n = 0
        if i > 0:
            prev_k = acc_K[i-1]
            prev_n = acc_N[i-1]

        conc_Y[prev_k:acc_K[i], prev_n:acc_N[i]] = Y

    return conc_Y


def transform_all_data(X, feature_assays):

    unq_assays = np.unique(feature_assays)
    assay_X = []
    assay_Y = []
    assay_labels = []

    print("Transform all data.")
    for unq_a in unq_assays:
        relevant_idx = (feature_assays == unq_a)
        relevant_X = X[:, relevant_idx]
        X_trans, pcs = transform_data(relevant_X)
        print("\t", unq_a, relevant_X.shape, "-->", X_trans.shape)
        assay_X.append(X_trans)
        assay_Y.append(pcs)
        assay_labels.append([unq_a]*X_trans.shape[1])

    conc_X = np.concatenate(assay_X, axis=1)
    conc_Y = concatenate_pcs(assay_Y)
    result_assays = np.concatenate(assay_labels)

    print("Concatenated X: ", conc_X.shape)
    print("Concatenated Y: ", conc_Y.shape)

    return conc_X, conc_Y, result_assays


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

    # Remove empty rows and columns
    Z_nomissing, row_key, col_key = remove_missing(Z, out_dim=20)
    sample_ids = sample_ids[row_key]
    sample_groups = sample_groups[row_key]
    feature_assays = feature_assays[col_key]
    feature_genes = feature_genes[col_key]

    # Remove the columns with least variance
    col_key = variance_filter(Z_nomissing, feature_assays, v_frac)
    Z_filtered = Z_nomissing[:,col_key]
    feature_assays = feature_assays[col_key]
    feature_genes = feature_genes[col_key]
    
    # Standardize the remaining columns
    Z_std, mu, sigma = standardize_columns(Z_filtered)
    
    # Perform concatenated PCA
    X, pcs, factor_assays = transform_all_data(Z_std, feature_assays) 

    # Output the transformed data and the 
    # fitted principal components and standardization parameters
    with h5py.File(trans_hdf, "w", driver="core") as f:
        su.write_hdf(f, "X", X.transpose())
        su.write_hdf(f, "instances", sample_ids, is_string=True) 
        su.write_hdf(f, "instance_groups", sample_groups, is_string=True)
        su.write_hdf(f, "target", target, is_string=True) 
        su.write_hdf(f, "factor_assays", factor_assays, is_string=True) 
    
    with h5py.File(fitted_hdf, "w", driver="core") as f:
        su.write_hdf(f, "Y", pcs)
        su.write_hdf(f, "mu", mu)
        su.write_hdf(f, "sigma", sigma)
        su.write_hdf(f, "feature_assays", feature_assays, is_string=True) 
        su.write_hdf(f, "feature_genes", feature_genes, is_string=True) 
        su.write_hdf(f, "factor_assays", factor_assays, is_string=True) 
        

