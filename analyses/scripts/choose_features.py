
import script_util as su
import numpy as np
import argparse
import h5py
import json


def choose_features(all_features, feature_params): 

    if "dropomic" in feature_params.keys():

        dropped_omic_types = feature_params["dropomic"]
        dropped = set(dropped_omic_types)

        omic_type_ls = [feat.split("_")[-1] for feat in all_features]
        kept_feature_idx = [i+1 for i, ot in enumerate(omic_type_ls) if ot not in dropped] 

        return kept_feature_idx

    else:
        raise ValueError 


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_hdf")
    parser.add_argument("used_idx_json")
    parser.add_argument("--feature-str")

    args = parser.parse_args()

    all_features = su.load_features(args.dataset_hdf)    

    feature_params = su.parse_arg_str(args.feature_str)

    used_feature_idx = choose_features(all_features, 
                                       feature_params)

    with open(args.used_idx_json, "w") as f:
        json.dump(used_feature_idx, f)



