
from sklearn.metrics import average_precision_score, silhouette_score
import script_util as su
import numpy as np
import argparse
import json


def score_Y(true_Y, fitted_Y):

    scores = {}

    print("TRUE Y SHAPE:", true_Y.shape)
    print("FITTED Y SHAPE:", fitted_Y.shape)

    # average cosine similarity
    true_norm_sq = np.sum(true_Y*true_Y, axis=0)
    pred_norm_sq = np.sum(fitted_Y*fitted_Y, axis=0)
    similarities = np.sum(true_Y*fitted_Y, axis=0) / np.sqrt(true_norm_sq*pred_norm_sq)
    avg_sim = np.mean(similarities)
    scores["Y_mean_cos_sim"] = avg_sim

    # average precision for identifying non-zero factor components
    true_Y_vec = (np.reshape(true_Y, -1) != 0)
    pred_Y_vec = np.abs(np.reshape(fitted_Y, -1))
    combined_av_prec = average_precision_score(true_Y_vec, pred_Y_vec)
    scores["Y_combined_av_prec"] = combined_av_prec

    # average precision, averaged across factors
    true_binary = (true_Y != 0)
    fitted_score = np.abs(fitted_Y)
    precs = [average_precision_score(true_binary[:,k],fitted_score[:,k]) for k in range(true_Y.shape[1])]
    mean_av_prec = np.mean(precs)
    scores["Y_mean_av_prec"] = mean_av_prec

    return scores


def score_X(true_X, fitted_X, sample_groups):

    scores = {}
    
    print("TRUE X SHAPE:", true_X.shape)
    print("FITTED X SHAPE:", fitted_X.shape)
    ## Scores for accuracy
    # Average spearman
    # Whole-matrix spearman

    ## Scores for agreement with regularization
    # Silhouette score
    sil = silhouette_score(fitted_X, sample_groups, metric="euclidean")
    scores["X_silhouette"] = sil

    return scores



def compute_scores(true_hdf, fitted_hdf):
   
    scores = {}

    # Score the fit of X
    true_X = su.load_embedding(true_hdf)
    fitted_X = su.load_embedding(fitted_hdf)
    sample_groups = su.load_sample_groups(fitted_hdf)
    X_scores = score_X(true_X, fitted_X, sample_groups)
    scores.update(X_scores)

    # Score the fit of Y
    true_Y = su.load_feature_factors(true_hdf)
    fitted_Y = su.load_feature_factors(fitted_hdf)
    Y_scores = score_Y(true_Y, fitted_Y)
    scores.update(Y_scores)

    return scores



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("true_hdf")
    parser.add_argument("fitted_hdf")
    parser.add_argument("output_json")

    args = parser.parse_args()

    score_dict = compute_scores(args.true_hdf, args.fitted_hdf)

    with open(args.output_json, "w") as f:
        json.dump(score_dict, f)


if __name__=="__main__":

    main()


