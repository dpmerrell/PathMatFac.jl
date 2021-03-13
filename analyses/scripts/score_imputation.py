
from sklearn.metrics import average_precision_score, r2_score
import script_util as su
import numpy as np
import argparse 
import h5py
import json

def logistic(x):
    x = np.array(x)
    return 1.0/(1.0 + np.exp(-x))


def score_imputation(true_data, true_features, true_instances, 
                     pred_feat_factor, pred_inst_factor,
                     pred_features, pred_instances, mask):

    # Find the instances and features shared by
    # the true and predicted datasets
    true_feat_idx, pred_feat_idx = su.keymatch(true_features, pred_features)
    true_inst_idx, pred_inst_idx = su.keymatch(true_instances, pred_instances)

    true_data = true_data.transpose()
    # Exclude the instances and features not shared by 
    # the true and predicted datasets 
    print("TRUE INST IDX: ", len(true_inst_idx)) 
    print("\tmin: ", min(true_inst_idx))
    print("\tmax: ", max(true_inst_idx))
    print("TRUE FEAT IDX: ", len(true_feat_idx)) 
    print("\tmin: ", min(true_feat_idx)) 
    print("\tmax: ", max(true_feat_idx)) 
    print("TRUE DATA: ", true_data.shape) 
    true_inst_idx = np.array(true_inst_idx)
    true_feat_idx = np.array(true_feat_idx)
    true_data = true_data[true_inst_idx[:,np.newaxis], true_feat_idx[np.newaxis,:]] 
    true_features = true_features[true_feat_idx]
    true_instances = true_instances[true_inst_idx]

    pred_feat_factor = pred_feat_factor[:, pred_feat_idx]
    pred_features = pred_features[pred_feat_idx]
    pred_inst_factor = pred_inst_factor[:, pred_inst_idx]
    pred_instances = pred_instances[pred_inst_idx]

    # Create maps from names to idxs
    true_feat_to_idx = su.value_to_idx(true_features)
    true_inst_to_idx = su.value_to_idx(true_instances)

    pred_feat_to_idx = su.value_to_idx(pred_features)
    pred_inst_to_idx = su.value_to_idx(pred_instances)

    # Exclude coords from the mask that are not 
    # shared by true and pred datasets
    mask = [coord for coord in mask if (coord[0] in true_inst_to_idx.keys() and coord[1] in true_feat_to_idx.keys())]

    true_vals = [true_data[true_inst_to_idx[coord[0]], true_feat_to_idx[coord[1]]] for coord in mask]
    pred_prods = [np.dot(pred_inst_factor[:, pred_inst_to_idx[coord[0]]], pred_feat_factor[:, pred_feat_to_idx[coord[1]]]) for coord in mask]

    # Separate the "regressed" and "classified" missing data
    feature_losses = [su.feature_to_loss(coord[1]) for coord in mask]
    
    logistic_true = [val for (val, loss) in zip(true_vals, feature_losses) if loss=="logistic"]
    linear_true = [prod for (prod, loss) in zip(true_vals, feature_losses) if loss=="linear"]
    
    logistic_pred = [prod for (prod, loss) in zip(pred_prods, feature_losses) if loss=="logistic"]
    logistic_pred = logistic(logistic_pred)
    linear_pred = [prod for (prod, loss) in zip(pred_prods, feature_losses) if loss=="linear"]

    # Score the predictions against truth!
    linear_r2 = r2_score(linear_true, linear_pred)
    logistic_aucpr = average_precision_score(logistic_true, logistic_pred)

    score_dict = {"n_r2": len(linear_true),
                  "r2": linear_r2, 
                  "n_aucpr": len(logistic_true),
                  "aucpr": logistic_aucpr
                  }

    return score_dict

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Score the agreement between imputed data and true (held-out) data.")
    parser.add_argument("inferred_hdf", help="path to HDF file containing inferred factors")
    parser.add_argument("truth_hdf", help="path to HDF file containing true factors")
    parser.add_argument("mask_json", help="mask to JSON containing missing data mask")
    parser.add_argument("score_json", help="path to output JSON")

    args = parser.parse_args()

    true_data, true_instances, true_features = su.load_true(args.truth_hdf)

    pred_feat_factor, pred_inst_factor,\
    pred_features, pred_instances, pred_pwys = su.load_factor_hdf(args.inferred_hdf,
                                                    inst_key="instances",
                                                    feat_key="features")

    mask = json.load(open(args.mask_json, "r"))

    score_dict = score_imputation(true_data,
                                  true_features, true_instances, 
                                  pred_feat_factor, pred_inst_factor,
                                  pred_features, pred_instances,
                                  mask)

    with open(args.score_json, "w") as f:
        json.dump(score_dict, f)


