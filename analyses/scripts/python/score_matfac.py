
from sklearn.metrics import average_precision_score, silhouette_score
from scipy.stats import spearmanr
import script_util as su
import numpy as np
import argparse
import json


def compute_scores(true_hdf, fitted_hdf):
    pass


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("true_hdf")
    parser.add_argument("fitted_hdf")
    parser.add_argument("output_json")

    args = parser.parse_args()

    score_dict = compute_scores(args.true_hdf, args.fitted_hdf)

    with open(args.output_json, "w") as f:
        json.dump(score_dict, f)



