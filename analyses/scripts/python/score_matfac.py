

from sklearn.metrics import average_precision_score, roc_auc_score, r2_score
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import linear_sum_assignment
import script_util as su
import numpy as np
import argparse
import h5py
import json


def score_col_param(v_true, v_fitted, score_fn):
    return score_fn(v_true, v_fitted)    

def score_batch_param(true_values, fitted_values, score_fn):
    #scores = [r2_score(v_true, v_fitted, multioutput="uniform_average") for (v_true, v_fitted) in zip(true_values, fitted_values)]
    scores = [score_fn(v_true, v_fitted) for (v_true, v_fitted) in zip(true_values, fitted_values)]
    sizes = [np.prod(v.shape) for v in true_values]
    return np.dot(scores, sizes)/sum(sizes)


##########################################################
# Scores for the factors X,Y 
##########################################################
def span_similarity(Y_true, Y_fitted):
    # Compute the numerator
    YtYf = np.dot(Y_true, Y_fitted.transpose())
    numerator = np.sum(YtYf*YtYf)

    # Compute the denominator
    ut, st, vht = np.linalg.svd(Y_true)
    uf, sf, vhf = np.linalg.svd(Y_fitted)

    k_min = min(Y_true.shape[0], Y_fitted.shape[0])
    st = st[:k_min]
    sf = sf[:k_min]
    denom = np.sum(st*st*sf*sf)
    return numerator/denom

    #YfYf = np.dot(Y_fitted, Y_fitted.transpose())

    # Subtle point: We don't want to artificially penalize Y_fitted
    # for having fewer rows than Y_true. So we only use
    # Y_true's first K_fitted singular values in that case.
    #K_true = Y_true.shape[0]
    #K_fitted = Y_fitted.shape[0]
    #YtYt_sqfrobenius = None
    #if K_true > K_fitted:
    #    u, s, vh = np.linalg.svd(Y_true)
    #    YtYt_sqfrobenius = np.sum(s[:K_fitted]*s[:K_fitted])
    #else:
    #    YtYt = np.dot(Y_true, Y_true.transpose())
    #    YtYt_sqfrobenius = np.sum(YtYt*YtYt)

    #return np.sum(YtYf*YtYf) / np.sqrt(YtYt_sqfrobenius*np.sum(YfYf*YfYf))


# Compute a simple p-value for this cosine similarity
# by repeatedly shuffling each row of Y_fitted.
# Return the average value under the null, and the p-value for 
# the given test_score. 
def spansim_pvalue(Y_true, Y_fitted, test_score, n_samples=10000):
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
        spansim = span_similarity(Y_true, Y_random)
        total_score += spansim
        if spansim >= test_score:
            count += 1

    return total_score/n_samples, count/n_samples


##########################################################
# Scores for the assignment matrix (A)
##########################################################

def rowwise_sq_cossim(Y_true, Y_fitted):
    
    K_true = Y_true.shape[0]
    Y_true_norm = Y_true / np.sqrt(np.sum(Y_true*Y_true, axis=1)).reshape(K_true,1)
    
    K_fitted = Y_fitted.shape[0]
    Y_fitted_norm = Y_fitted / np.sqrt(np.sum(Y_fitted*Y_fitted, axis=1)).reshape(K_fitted,1)
    cossim_matrix = Y_true_norm @ np.transpose(Y_fitted_norm)
    return cossim_matrix*cossim_matrix 


def choose_best_match(Y_true, Y_fitted):
    
    score_matrix = rowwise_sq_cossim(Y_true, Y_fitted) 
    
    true_idx, fitted_idx = linear_sum_assignment(score_matrix, maximize=True)
    score_ls = [score_matrix[i,j] for (i,j) in zip(true_idx, fitted_idx)]
    mean_sq_cossim = np.mean(score_ls)

    return true_idx, fitted_idx, mean_sq_cossim


def colwise_score(A_true, A_fitted, score_fn):
    K = A_true.shape[1]
    score_vec = np.zeros(K)
    for k in range(K):
        score_vec[k] = score_fn(A_true[:,k], A_fitted[:,k])
    return np.mean(score_vec)


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


def load_A_matrices(hfile):
    k_ls = sorted(list(hfile["fsard/A"].keys()))
    print("K LIST")
    print(k_ls)
    return [hfile["fsard/A/"+k][:,:].transpose() for k in k_ls]


def compute_scores(true_hdf, fitted_hdf):

    scores = dict()

    with h5py.File(true_hdf, "r") as f_true:
        with h5py.File(fitted_hdf, "r") as f_fitted:

            print("Scoring Y...")
            Y_true = f_true["Y"][:,:].transpose()
            Y_fitted = f_fitted["Y"][:,:].transpose()
            scores["Y_spansim"] = span_similarity(Y_true, Y_fitted)
            #spansim_null, spansim_p_value = spansim_pvalue(Y_true, Y_fitted, scores["Y_spansim"])
            #scores["Y_spansim_null"] = spansim_null
            #scores["Y_spansim_pvalue"] = spansim_p_value

            print("Scoring X...")
            X_true = f_true["X"][:,:].transpose()
            X_fitted = f_fitted["X"][:,:].transpose()
            scores["X_spansim"] = span_similarity(X_true, X_fitted)
            #spansim_null, spansim_p_value = spansim_pvalue(X_true, X_fitted, scores["X_spansim"])
            #scores["X_spansim_null"] = spansim_null
            #scores["X_spansim_pvalue"] = spansim_p_value

            if ("fsard" in f_true.keys()) and ("fsard" in f_fitted.keys()): 
                print("Scoring A...")
                
                print("Matching true and fitted factors:")
                true_idx, fitted_idx, sq_cossim = choose_best_match(Y_true, Y_fitted)
                print("True idx:", true_idx)
                print("Fitted idx:", fitted_idx)
                print("RMS cosine similarity:", np.sqrt(sq_cossim))
                scores["Y_best_rms_cossim"] = np.sqrt(sq_cossim)

                A_true_ls = load_A_matrices(f_true) 
                A_fitted_ls = load_A_matrices(f_fitted)

                A_true_ls = [At[:,true_idx] for At in A_true_ls]
                A_fitted_ls = [Af[:,fitted_idx] for Af in A_fitted_ls]

                aucroc_fn = lambda at, af: safe_aucroc((at>0).astype(int), af)
                scores["A_aucroc"] = np.mean([colwise_score(At, Af, aucroc_fn) for (At,Af) in zip(A_true_ls, A_fitted_ls)])
                
                aucpr_fn = lambda at, af: average_precision_score((at>0).astype(int), af)
                scores["A_aucpr"] = np.mean([colwise_score(At, Af, aucpr_fn) for (At,Af) in zip(A_true_ls, A_fitted_ls)])

                scores["A_aucpr_baserate"] = np.mean([np.mean(A_true[:,j] > 0) for A_true in A_true_ls for j in range(A_true.shape[1])])


            
            print("Scoring mu...")
            mu_true = f_true["mu"][:]
            mu_fitted = f_fitted["mu"][:]
            scores["mu_r2"] = score_col_param(mu_true, mu_fitted, r2_score)

            print("Scoring logsigma...")
            logsigma_true = f_true["logsigma"][:]
            logsigma_fitted = f_fitted["logsigma"][:]
            scores["logsigma_r2"] = score_col_param(logsigma_true, logsigma_fitted, r2_score)
            scores["logsigma_spearman"] = score_col_param(logsigma_true, logsigma_fitted, lambda mt,mf: spearmanr(mt,mf)[0])
            scores["logsigma_pearson"] = score_col_param(logsigma_true, logsigma_fitted, lambda mt,mf: pearsonr(mt,mf)[0])

            if ("theta" in f_fitted.keys()) and ("theta" in f_true.keys()):
                print("Scoring batch parameters...")
                n_batches = len([k for k in f_true["theta"].keys() if k.startswith("values_")])
                theta_true = [f_true[f"theta/values_{k+1}"][:,:] for k in range(n_batches)]
                theta_true = [v.flatten() for v in theta_true]
                theta_fitted = [f_fitted[f"theta/values_{k+1}"][:,:] for k in range(n_batches)]
                theta_fitted = [v.flatten() for v in theta_fitted]

                batch_r2 = lambda tr,fit: r2_score(tr,fit)
                batch_pearson = lambda tr,fit: pearsonr(tr,fit)[0]
                batch_spearman = lambda tr,fit: spearmanr(tr,fit)[0]

                scores["theta_r2"] = score_batch_param(theta_true, theta_fitted, batch_r2)
                scores["theta_pearson"] = score_batch_param(theta_true, theta_fitted, batch_pearson)
                scores["theta_spearman"] = score_batch_param(theta_true, theta_fitted, batch_spearman)
                
                logdelta_true = [f_true[f"logdelta/values_{k+1}"][:,:] for k in range(n_batches)]
                logdelta_true = [v.flatten() for v in logdelta_true]
                logdelta_fitted = [f_fitted[f"logdelta/values_{k+1}"][:,:] for k in range(n_batches)]
                logdelta_fitted = [v.flatten() for v in logdelta_fitted]
                for ldf in logdelta_fitted:
                    ldf[np.logical_not(np.isfinite(ldf))] = 0

                scores["logdelta_r2"] = score_batch_param(logdelta_true, logdelta_fitted, batch_r2)
                scores["logdelta_pearson"] = score_batch_param(logdelta_true, logdelta_fitted, batch_pearson)
                scores["logdelta_spearman"] = score_batch_param(logdelta_true, logdelta_fitted, batch_spearman)

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



