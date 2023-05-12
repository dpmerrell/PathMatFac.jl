
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
  

def standardize_columns(Z):

    col_std = np.nanstd(Z, axis=0)
    col_means = np.nanmean(Z, axis=0)

    Z_std = (Z - col_means.transpose()) / col_std.transpose()

    return Z_std, col_means, col_std 


def find_elbow(rsquare):

    # Find maximum of discrete 2nd derivative
    d1 = rsquare[1:] - rsquare[:-1]
    d2 = d1[1:] - d1[:-1]
    
    # Off-by-one because of finite difference
    max_idx = np.argmax(d2) + 1
    return max_idx


def compute_principal_components(Z, min_components=2, max_components=20):

    result = PCA(Z, standardize=False,
                    method='nipals',
                    demean=False,
                    normalize=False,
                    ncomp=max_components, 
                    missing="fill-em",
                    max_em_iter=5)

    rsquare = result.rsquare
    pcs = result.coeff

    elbow_idx = find_elbow(rsquare)
    elbow_idx = max(min_components, elbow_idx)

    pcs = pcs[:elbow_idx,:]

    return pcs
    

def compute_assay_pcs(X, feature_assays, max_components=20):

    unq_assays = np.unique(feature_assays)

    all_pcs = []

    print("Computing assay-specific principal components...")
    for ua in unq_assays:
        rel_idx = np.where(feature_assays == ua)[0]
        rel_X = X[:,rel_idx]

        kept_rows = nan_filter(rel_X, count_axis=1, min_count=max_components)
        rel_X = rel_X[kept_rows, :]

        pcs = compute_principal_components(rel_X, max_components=max_components)
        
        print("\t",ua,":", pcs.shape)

        all_pcs.append(pcs)

    return all_pcs 


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
                                 min_finite_count=20, var_filter_frac=v_frac)
    print("Column filtering: ", Z.shape[1], "-->", len(keep_idx))
    Z = Z[:,keep_idx]
    feature_assays = feature_assays[keep_idx]
    feature_genes = feature_genes[keep_idx]

    # Standardize the data 
    Z_std, mu, sigma = standardize_columns(Z)

    print("Z_std:", Z_std.shape)

    # principal components for individual assays
    all_pcs = compute_assay_pcs(Z_std, feature_assays)

    # Concatenate the assay-wise results
    full_pcs = concatenate_pcs(all_pcs)


    # Transform standardized data via concatenated principal components
    X = su.linear_transform(Z_std, full_pcs, max_iter=5000, rel_tol=1e-6)
    
    print("X:", X.shape)
    print("Y:", full_pcs.shape)

    # Output the transformed data and the 
    # fitted principal components and standardization parameters
    with h5py.File(trans_hdf, "w", driver="core") as f:
        su.write_hdf(f, "X", X)
        su.write_hdf(f, "instances", sample_ids, is_string=True) 
        su.write_hdf(f, "instance_groups", sample_groups, is_string=True)
        su.write_hdf(f, "target", target, is_string=True) 
    
    with h5py.File(fitted_hdf, "w", driver="core") as f:
        su.write_hdf(f, "Y", full_pcs.transpose())
        su.write_hdf(f, "mu", mu)
        su.write_hdf(f, "sigma", sigma)
        su.write_hdf(f, "feature_assays", feature_assays, is_string=True) 
        su.write_hdf(f, "feature_genes", feature_genes, is_string=True) 
        

