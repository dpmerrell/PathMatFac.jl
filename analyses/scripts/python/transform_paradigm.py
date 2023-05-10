# transform_paradigm.py
#
# Fetch precomputed PARADIGM outputs for the given
# TCGA dataset. Then use PCA to reduce the dimensionality.

import script_util as su
import numpy as np
import argparse
import h5py


def load_data(data_df, paradigm_hdf):

    with h5py.File(data_hdf, "r") as f:
        sample_ids = f["omic_data/instances"][:].astype(str)
        sample_groups = f["omic_data/instance_groups"][:].astype(str)
        target = f["target"][...].transpose().astype(str)

    with h5py.File(paradigm_hdf, "r") as f:
        paradigm_data = f["data"][:,:].transpose()
        paradigm_features = f["features"][:].astype(str)
        paradigm_samples = f["instances"][:].astype(str)

        left_idx, right_idx = su.keymatch(sample_ids, paradigm_samples)

        sample_ids = sample_ids[left_idx]
        target = target[left_idx,...]
        sample_groups = sample_groups[left_idx]
        X = paradigm_data[right_idx,:]

    return X, sample_ids, sample_groups, paradigm_features, target



def load_model(pc_hdf):

    Y = su.load_hdf(pc_hdf, "Y").transpose()
    mu = su.load_hdf(pc_hdf, "mu")
    sigma = su.load_hdf(pc_hdf, "sigma")

    features = su.load_hdf(pc_hdf, "features", dtype=str)

    return Y, mu, sigma, features


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("data_hdf")
    parser.add_argument("paradigm_hdf")
    parser.add_argument("pc_hdf")
    parser.add_argument("transformed_data_hdf")
    args = parser.parse_args()

    data_hdf = args.data_hdf
    paradigm_hdf = args.paradigm_hdf
    pc_hdf = args.pc_hdf
    out_hdf = args.transformed_data_hdf

    # Load the data and model parameters
    Z_test, sample_ids, sample_groups, test_features, target = load_data(data_hdf, paradigm_hdf)
    Y, mu, sigma, model_features = load_model(pc_hdf)

    # Find the intersection of features that are (1) in the test data
    # and (2) covered by the model
    data_col_idx, model_col_idx = su.keymatch(test_features, model_features)

    # Filter the data and model params to keep only those features
    Z_test = Z_test[:,data_col_idx]
    test_features = test_features[data_col_idx]

    Y = Y[:,model_col_idx]
    mu = mu[model_col_idx]
    sigma = sigma[model_col_idx]
    model_features = model_features[model_col_idx]

    # Standardize the test data
    Z_std = (Z_test - mu) / sigma

    # Use the principal components to transform the data
    # (have to handle missing data)
    X = su.linear_transform(Z_std, Y)

    # Save the transformed data to HDF
    with h5py.File(out_hdf, "w", driver="core") as f:
        su.write_hdf(f, "X", X)
        su.write_hdf(f, "instances", sample_ids, is_string=True) 
        su.write_hdf(f, "instance_groups", sample_groups, is_string=True)
        su.write_hdf(f, "target", target.transpose(), is_string=True)


