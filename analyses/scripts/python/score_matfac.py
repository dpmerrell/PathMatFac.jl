

from sklearn.metrics import average_precision_score, roc_auc_score, r2_score
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment
import script_util as su
import numpy as np
import argparse
import h5py
import json

def score_col_param(v_true, v_fitted):
    return r2_score(v_true, v_fitted)    


def score_batch_param(true_values, fitted_values):
    scores = [r2_score(v_true, v_fitted, multioutput="uniform_average") for (v_true, v_fitted) in zip(true_values, fitted_values)]
    sizes = [np.prod(v.shape) for v in true_values]
    print(scores)
    print(sizes)
    return np.dot(scores, sizes)/sum(sizes)


def generalized_cosine_similarity(Y_true, Y_fitted):
    return np.sum(np.dot(Y_true, Y_fitted.transpose())) / np.sqrt(np.sum(Y_true*Y_true)*np.sum(Y_fitted*Y_fitted))


def maximal_pairwise_score(A_true, A_fitted, score_fn):

    K_true = A_true.shape[1]
    K_fitted = A_fitted.shape[1]
    score_matrix = np.zeros((K_fitted, K_true))
    for i in range(K_fitted):
        for j in range(K_true):
            score_matrix[i,j] = score_fn(A_true[:,j], A_fitted[:,i])

    row_idx, col_idx = linear_sum_assignment(score_matrix, maximize=True)
    score_ls = [score_matrix[i,j] for (i,j) in zip(row_idx, col_idx)]

    return sum(score_ls)/len(score_ls)


def compute_scores(true_hdf, fitted_hdf):

    scores = dict()

    with h5py.File(true_hdf, "r") as f_true:
        with h5py.File(fitted_hdf, "r") as f_fitted:

            Y_true = f_true["Y"][:,:].transpose()
            Y_fitted = f_fitted["Y"][:,:].transpose()
            scores["Y_cossim"] = generalized_cosine_similarity(Y_true, Y_fitted)
           
            if ("fsard" in f_true.keys()) and ("fsard" in f_fitted.keys()): 
                A_true = f_true["fsard/A"][:,:].transpose()
                A_fitted = f_fitted["fsard/A"][:,:].transpose()
                scores["A_aucroc"] = maximal_pairwise_score(A_true, A_fitted, lambda at, af: roc_auc_score((at>0).astype(int), af))
                scores["A_aucpr"] = maximal_pairwise_score(A_true, A_fitted, lambda at, af: average_precision_score((at>0).astype(int), af))
                scores["A_aucpr_baseline"] = np.mean([np.mean(A_true[:,j] > 0) for j in range(A_true.shape[1])])

                S_true = f_true["fsard/S"][:,:].transpose()
                beta_true = np.matmul(A_true.transpose(), S_true)

                S_fitted = f_fitted["fsard/S"][:,:].transpose()
                beta_fitted = np.matmul(A_fitted.transpose(), S_fitted)
                scores["beta_aucroc"] = maximal_pairwise_score(beta_true, beta_fitted, lambda bt, bf: roc_auc_score((bt>0).astype(int), bf))

            mu_true = f_true["mu"][:]
            mu_fitted = f_fitted["mu"][:]
            scores["mu_r2"] = score_col_param(mu_true, mu_fitted)

            logsigma_true = f_true["logsigma"][:]
            logsigma_fitted = f_fitted["logsigma"][:]
            scores["logsigma_r2"] = score_col_param(logsigma_true, logsigma_fitted)

            if ("theta" in f_true.keys()):
                n_batches = len([k for k in f_true["theta"].keys() if k.startswith("values_")])
                theta_true = [f_true[f"theta/values_{k+1}"][:,:] for k in range(n_batches)]
                theta_fitted = [f_fitted[f"theta/values_{k+1}"][:,:] for k in range(n_batches)]
                scores["theta_r2"] = score_batch_param(theta_true, theta_fitted)
                
                logdelta_true = [f_true[f"logdelta/values_{k+1}"][:,:] for k in range(n_batches)]
                logdelta_fitted = [f_fitted[f"logdelta/values_{k+1}"][:,:] for k in range(n_batches)]
                scores["logdelta_r2"] = score_batch_param(logdelta_true, logdelta_fitted)

    return scores


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("true_hdf")
    parser.add_argument("fitted_hdf")
    parser.add_argument("output_json")

    args = parser.parse_args()

    score_dict = compute_scores(args.true_hdf, args.fitted_hdf)

    with open(args.output_json, "w") as f:
        json.dump(score_dict, f)



