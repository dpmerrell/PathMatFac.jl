
import script_util as su
import numpy as np
import argparse
import h5py
import json


def drop_features(all_features, dropped_omic_types): 

    dropped = set(dropped_omic_types)

    omic_type_ls = [feat.split("_")[-1] for feat in all_features]
    dropped_feature_idx = [i+1 for i, ot in enumerate(omic_type_ls) if ot not in dropped] 

    return dropped_feature_idx 


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_hdf")
    parser.add_argument("used_idx_json")
    parser.add_argument("--dropped-omic-types", nargs="*")

    args = parser.parse_args()

    all_features = su.load_features(args.dataset_hdf)    

    used_feature_idx = drop_features(all_features, 
                                     args.dropped_omic_types)

    with open(args.used_idx_json, "w") as f:
        json.dump(used_feature_idx, f)



