
from sklearn.metrics import average_precision_score, r2_score
import script_util as su
import numpy as np
import argparse 
import h5py
import json


def score_imputation(true_data, true_features, true_instances, 
                     pred_values, mask):

    ## Find the instances and features shared by
    ## the true and predicted datasets
    #true_feat_idx, pred_feat_idx = su.keymatch(true_features, pred_features)
    #true_inst_idx, pred_inst_idx = su.keymatch(true_instances, pred_instances)

    #true_data = true_data.transpose()
    ## Exclude the instances and features not shared by 
    ## the true and predicted datasets 
    #print("TRUE INST IDX: ", len(true_inst_idx)) 
    #print("\tmin: ", min(true_inst_idx))
    #print("\tmax: ", max(true_inst_idx))
    #print("TRUE FEAT IDX: ", len(true_feat_idx)) 
    #print("\tmin: ", min(true_feat_idx)) 
    #print("\tmax: ", max(true_feat_idx)) 
    #print("TRUE DATA: ", true_data.shape) 
    #true_inst_idx = np.array(true_inst_idx)
    #true_feat_idx = np.array(true_feat_idx)
    #true_data = true_data[true_inst_idx[:,np.newaxis], true_feat_idx[np.newaxis,:]] 
    #true_features = true_features[true_feat_idx]
    #true_instances = true_instances[true_inst_idx]

    # Create maps from names to idxs
    true_feat_to_idx = su.value_to_idx(true_features)
    true_inst_to_idx = su.value_to_idx(true_instances)

    # Exclude coords from the mask that are not 
    # shared by true and pred datasets
    kept_idx = [idx for idx, v in enumerate(pred_values) if v is not None]
    kept_mask = [mask[idx] for idx in kept_idx]
    kept_pred = [pred_values[idx] for idx in kept_idx] 

    true_vals = [true_data[true_feat_to_idx[coord[1]], true_inst_to_idx[coord[0]]] for coord in kept_mask]
    print("TRUE VALUES:", len(true_vals))#, true_vals)
    print("PRED VALUES:", len(kept_pred))#, kept_pred)

    # Separate the "regressed" and "classified" missing data
    feature_losses = [su.feature_to_loss(coord[1]) for coord in kept_mask]
    
    logistic_true = [val for (val, loss) in zip(true_vals, feature_losses) if loss=="logistic"]
    linear_true = [val for (val, loss) in zip(true_vals, feature_losses) if loss=="linear"]
    
    logistic_pred = [val for (val, loss) in zip(kept_pred, feature_losses) if loss=="logistic"]
    linear_pred = [val for (val, loss) in zip(kept_pred, feature_losses) if loss=="linear"]

    # Score the predictions against truth!
    linear_r2 = r2_score(linear_true, linear_pred)
    logistic_aucpr = average_precision_score(logistic_true, logistic_pred)

    score_dict = {"reg_n": len(linear_true),
                  "reg_r2": linear_r2, 
                  "cls_n": len(logistic_true),
                  "cls_aucpr": logistic_aucpr,
                  "cls_true_n1": len([tr for tr in logistic_true if tr == 1]),
                  }

    return score_dict

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Score the agreement between imputed data and true (held-out) data.")
    parser.add_argument("imputed_json", help="path to JSON file containing imputed values")
    parser.add_argument("truth_hdf", help="path to HDF file containing true factors")
    parser.add_argument("mask_json", help="mask to JSON containing missing data mask")
    parser.add_argument("score_json", help="path to output JSON")

    args = parser.parse_args()

    pred_values = json.load(open(args.imputed_json, "r"))

    true_data, true_instances, true_features = su.load_true(args.truth_hdf)

    mask = json.load(open(args.mask_json, "r"))

    score_dict = score_imputation(true_data,
                                  true_features, true_instances,
                                  pred_values, mask)

    with open(args.score_json, "w") as f:
        json.dump(score_dict, f)


