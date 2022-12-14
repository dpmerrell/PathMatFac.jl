
from statsmodels.multivariate.pca import PCA
import h5py
import sys

def load_data(data_hdf, modality="mrnaseq"):

    X = None

    with h5py.File(data_hdf, "r") as f:

        data = f["omic_data"]["data"][:,:].transpose()
        assays = f["omic_data"]["feature_assays"][:]

        relevant_cols = (assays == modality)
        X = data[:,relevant_cols]

    return X


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


def find_knee(rsquare):

    # Find maximum of discrete 2nd derivative
    d1 = rsquare[1:] - rsquare[:-1]
    d2 = d1[1:] - d1[:-1]
    
    # Off-by-one because of finite difference
    max_idx = np.argmax(d2) + 1
    return max_idx


def transform_data(X, n_components=20):

    result = PCA(X, standardize=True,
                    ncomp=n_components, 
                    missing="fill-em",
                    max_em_iter=500)

    X_trans = result.factors
    rsquare = result.rsquare

    knee_idx = find_knee(rsquare)
    X_trans = X_trans[:,:knee_idx]

    return X_trans


if __name__=="__main__":

    
    data_hdf = args[1]
    transformed_hdf = args[2]

    X = load_data(data_hdf)
    X_nomissing, row_key, col_key = remove_missing(X)

    

