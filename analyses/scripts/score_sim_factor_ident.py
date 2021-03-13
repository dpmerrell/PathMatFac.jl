
import script_util as su
import numpy as np
import argparse 
import h5py
import json


def score_inferred_factors(true_feat_factor, true_inst_factor,
                           true_features, true_instances, true_pwys,
                           pred_feat_factor, pred_inst_factor,
                           pred_features, pred_instances, pred_pwys):

    # Match the corresponding indices 
    # between true and predicted factors
    true_feat_idx, pred_feat_idx = su.keymatch(true_features, pred_features)
    true_inst_idx, pred_inst_idx = su.keymatch(true_instances, pred_instances)
    true_pwy_idx, pred_pwy_idx = su.keymatch(true_pwys, pred_pwys)

    print(true_pwys)
    print(pred_pwys)

    # Extract the matched features
    true_feat_factor = true_feat_factor[true_pwy_idx, true_feat_idx]
    pred_feat_factor = pred_feat_factor[pred_pwy_idx, pred_feat_idx]

    # Extract the matched instances
    true_inst_factor = true_inst_factor[true_pwy_idx, true_inst_idx]
    pred_inst_factor = pred_inst_factor[pred_pwy_idx, pred_inst_idx]

    print("FEATURE FACTOR SHAPES:", true_feat_factor.shape, pred_feat_factor.shape)
    print("PATIENT FACTOR SHAPES:", true_inst_factor.shape, pred_inst_factor.shape)



if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Score the agreement between inferred matrix factors and true (simulated) matrix factors.")
    parser.add_argument("inferred_hdf", help="path to HDF file containing inferred factors")
    parser.add_argument("truth_hdf", help="path to HDF file containing true factors")
    parser.add_argument("score_json", help="path to output JSON")

    args = parser.parse_args()

    true_feat_factor, true_inst_factor,\
    true_features, true_instances, true_pwys = su.load_factor_hdf(args.truth_hdf)

    pred_feat_factor, pred_inst_factor,\
    pred_features, pred_instances, pred_pwys = su.load_factor_hdf(args.inferred_hdf,
                                                    inst_key="instances",
                                                    feat_key="features")

    score_dict = score_inferred_factors(true_feat_factor, true_inst_factor,
                                        true_features, true_instances, true_pwys,
                                        pred_feat_factor, pred_inst_factor,
                                        pred_features, pred_instances, pred_pwys)

    with open(args.score_json, "w") as f:
        json.dump(score_dict, f)


