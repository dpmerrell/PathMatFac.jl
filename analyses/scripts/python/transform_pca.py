
import script_util as su
import numpy as np
import argparse
import h5py


def load_data(data_hdf):
    Z_test = su.load_hdf(data_hdf, "omic_data/data").transpose()
    feature_genes = su.load_hdf(data_hdf, "omic_data/feature_genes", dtype=str)
    feature_assays = su.load_hdf(data_hdf, "omic_data/feature_assays", dtype=str)
    instances = su.load_hdf(data_hdf, "omic_data/instances", dtype=str)
    instance_groups = su.load_hdf(data_hdf, "omic_data/instance_groups", dtype=str)
    target = su.load_hdf(data_hdf, "target", dtype=str)
 
    return Z_test, feature_genes, feature_assays, instances, instance_groups, target


def load_model(pc_hdf):

    Y = su.load_hdf(pc_hdf, "Y").transpose()
    mu = su.load_hdf(pc_hdf, "mu")
    sigma = su.load_hdf(pc_hdf, "sigma")

    feature_genes = su.load_hdf(pc_hdf, "feature_genes", dtype=str)
    feature_assays = su.load_hdf(pc_hdf, "feature_assays", dtype=str)

    return Y, mu, sigma, feature_genes, feature_assays


def select_features(test_genes, test_assays,
                    model_genes, model_assays):
    test_keys = ["_".join(pair) for pair in zip(test_genes, test_assays)] 
    model_keys = ["_".join(pair) for pair in zip(model_genes, model_assays)] 

    test_idx, model_idx = su.keymatch(test_keys, model_keys)

    return test_idx, model_idx
 

def pca_transform(Z, Y, lr=2.0, rel_tol=1e-8, max_iter=1000):

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
        if (lss - new_lss)/lss < rel_tol:
            print("Loss decrease < rel tol ({}). Terminating".format(rel_tol))
            break

        # Update loop variables
        lss = new_lss
        i += 1
        print("Iteration: {}; Loss: {:.2f}".format(i, lss))

    if i >= max_iter:
        print("Reached max iter ({}). Terminating".format(max_iter))

    return X


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("data_hdf")
    parser.add_argument("pc_hdf")
    parser.add_argument("transformed_data_hdf")
    args = parser.parse_args()

    data_hdf = args.data_hdf
    pc_hdf = args.pc_hdf
    out_hdf = args.transformed_data_hdf

    # Load the data and model parameters
    Z_test, test_genes, test_assays, sample_ids, sample_groups, target = load_data(data_hdf)
    Y, mu, sigma, model_genes, model_assays = load_model(pc_hdf)

    # Find the intersection of features that are (1) in the test data
    # and (2) covered by the model
    data_col_idx, model_col_idx = select_features(test_genes, test_assays, 
                                                  model_genes, model_assays)

    # Filter the data and model params to keep only those features
    Z_test = Z_test[:,data_col_idx]
    test_genes = test_genes[data_col_idx]
    test_assays = test_assays[data_col_idx]

    Y = Y[:,model_col_idx]
    mu = mu[model_col_idx]
    sigma = sigma[model_col_idx]
    model_genes = model_genes[model_col_idx]
    model_assays = model_assays[model_col_idx]

    # Standardize the test data
    Z_std = (Z_test - mu) / sigma

    # Use the principal components to transform the data
    # (have to handle missing data)
    X = pca_transform(Z_std, Y)

    # Save the transformed data to HDF
    with h5py.File(out_hdf, "w") as f:
        su.write_hdf(f, "X", X)
        su.write_hdf(f, "instances", sample_ids, is_string=True) 
        su.write_hdf(f, "instance_groups", sample_groups, is_string=True)
        su.write_hdf(f, "target", target, is_string=True)

