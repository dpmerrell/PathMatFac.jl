

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
    return np.dot(scores, sizes)/sum(sizes)


##########################################################
# Scores for the factors X,Y 
##########################################################
def generalized_cosine_similarity(Y_true, Y_fitted):
    ip_tf = np.dot(Y_true, Y_fitted.transpose())
    ip_tt = np.dot(Y_true, Y_true.transpose())
    ip_ff = np.dot(Y_fitted, Y_fitted.transpose())
    return np.sum(ip_tf*ip_tf) / np.sqrt(np.sum(ip_tt*ip_tt)*np.sum(ip_ff*ip_ff))


# Compute a simple p-value for this cosine similarity
# by repeatedly shuffling each row of Y_fitted.
# Return the average value under the null, and the p-value for 
# the given test_score. 
def generalized_cossim_pvalue(Y_true, Y_fitted, test_score, n_samples=10000):
    # Generate random matrices similar to Y_fitted 
    K, N = Y_fitted.shape
    #means = np.mean(Y_fitted, axis=1).reshape(K,1)
    #stds = np.std(Y_fitted, axis=1).reshape(K,1)

    count = 0
    total_score = 0.0
    Y_random = Y_fitted.copy()
    perm = np.random.permutation(N)

    print("Computing p-value for cosine similarity")

    for i in range(n_samples):
        perm[:] = np.random.permutation(N)
        Y_random[:,:] = Y_fitted[:,perm]
        cossim = generalized_cosine_similarity(Y_true, Y_random)
        total_score += cossim
        if cossim >= test_score:
            count += 1

    return total_score/n_samples, count/n_samples


##########################################################
# Scores for the assignment matrix (A)
##########################################################
def maximal_pairwise_score(A_true, A_fitted, score_fn, verbose=False):

    K_true = A_true.shape[1]
    K_fitted = A_fitted.shape[1]
    score_matrix = np.zeros((K_fitted, K_true))
    for i in range(K_fitted):
        for j in range(K_true):
            score_matrix[i,j] = score_fn(A_true[:,j], A_fitted[:,i])

    row_idx, col_idx = linear_sum_assignment(score_matrix, maximize=True)
    score_ls = [score_matrix[i,j] for (i,j) in zip(row_idx, col_idx)]

    if verbose:
        print("Matched factors:")
        print("True idx:", col_idx)
        print("Fitted idx:", row_idx)

    return sum(score_ls)/len(score_ls)


def mps_pvalue(A_true, A_fitted, score_fn, test_score, n_samples=10000):

    L,K = A_fitted.shape

    count = 0
    total_score = 0.0
    A_random = A_fitted.copy()
    perm = np.random.permutation(L)
   
    print("Computing p-value for ", score_fn)
 
    for i in range(n_samples):
        perm[:] = np.random.permutation(L)
        A_random = A_fitted[perm,:]
        sc = maximal_pairwise_score(A_true, A_random, score_fn)
        total_score += sc
        if sc >= test_score:
            count += 1

    return total_score/n_samples, count/n_samples


def safe_aucroc(y_true, y_pred):
    if len(np.unique(y_true)) == 1:
        return 0.5
    else:
        return roc_auc_score(y_true, y_pred)


def compute_scores(true_hdf, fitted_hdf):

    scores = dict()

    with h5py.File(true_hdf, "r") as f_true:
        with h5py.File(fitted_hdf, "r") as f_fitted:

            print("Scoring Y...")
            Y_true = f_true["Y"][:,:].transpose()
            Y_fitted = f_fitted["Y"][:,:].transpose()
            scores["Y_cossim"] = generalized_cosine_similarity(Y_true, Y_fitted)
            cossim_null, cossim_p_value = generalized_cossim_pvalue(Y_true, Y_fitted, scores["Y_cossim"])
            scores["Y_cossim_null"] = cossim_null
            scores["Y_cossim_pvalue"] = cossim_p_value

            print("Scoring X...")
            X_true = f_true["X"][:,:].transpose()
            X_fitted = f_fitted["X"][:,:].transpose()
            scores["X_cossim"] = generalized_cosine_similarity(X_true, X_fitted)
            cossim_null, cossim_p_value = generalized_cossim_pvalue(X_true, X_fitted, scores["X_cossim"])
            scores["X_cossim_null"] = cossim_null
            scores["X_cossim_pvalue"] = cossim_p_value

            if ("fsard" in f_true.keys()) and ("fsard" in f_fitted.keys()): 
                print("Scoring A...")
                A_true = f_true["fsard/A"][:,:].transpose()
                A_fitted = f_fitted["fsard/A"][:,:].transpose()

                aucroc_fn = lambda at, af: safe_aucroc((at>0).astype(int), af)
                scores["A_aucroc"] = maximal_pairwise_score(A_true, A_fitted, aucroc_fn, verbose=True)
                A_aucroc_null, A_aucroc_pvalue = mps_pvalue(A_true, A_fitted, aucroc_fn, scores["A_aucroc"])
                scores["A_aucroc_null"] = A_aucroc_null
                scores["A_aucroc_pvalue"] = A_aucroc_pvalue
                
                aucpr_fn = lambda at, af: average_precision_score((at>0).astype(int), af)
                scores["A_aucpr"] = maximal_pairwise_score(A_true, A_fitted, aucpr_fn, verbose=True)
                A_aucpr_null, A_aucpr_pvalue = mps_pvalue(A_true, A_fitted, aucpr_fn, scores["A_aucpr"])
                scores["A_aucpr_null"] = A_aucpr_null
                scores["A_aucpr_pvalue"] = A_aucpr_pvalue
                scores["A_aucpr_baserate"] = np.mean([np.mean(A_true[:,j] > 0) for j in range(A_true.shape[1])])

                print("Scoring beta...")
                S_true = f_true["fsard/S"][:,:].transpose()
                beta_true = np.matmul(A_true.transpose(), S_true)

                S_fitted = f_fitted["fsard/S"][:,:].transpose()
                beta_fitted = np.matmul(A_fitted.transpose(), S_fitted)
                scores["beta_aucroc"] = maximal_pairwise_score(beta_true.transpose(), 
                                                               beta_fitted.transpose(), 
                                                               lambda bt, bf: safe_aucroc((bt>0).astype(int), bf),
                                                               verbose=True)

            
            print("Scoring mu...")
            mu_true = f_true["mu"][:]
            mu_fitted = f_fitted["mu"][:]
            scores["mu_r2"] = score_col_param(mu_true, mu_fitted)

            print("Scoring logsigma...")
            logsigma_true = f_true["logsigma"][:]
            logsigma_fitted = f_fitted["logsigma"][:]
            scores["logsigma_r2"] = score_col_param(logsigma_true, logsigma_fitted)

            if ("theta" in f_true.keys()):
                print("Scoring batch parameters...")
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



