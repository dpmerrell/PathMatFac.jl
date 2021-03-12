
import script_util as su
import numpy as np
import argparse 
import h5py
import json


def load_factor_hdf(factor_hdf, feat_key="index", inst_key="columns"):

    with h5py.File(factor_hdf, "r") as f:
        feat_factor = f["feature_factor"][:,:]
        inst_factor = f["instance_factor"][:,:]
        features = f[feat_key][:].astype(str)
        instances = f[inst_key][:].astype(str)
        pathways = f["pathways"][:].astype(str)

    pathways = np.array([pwy.split("/")[-1] for pwy in pathways])

    print("SHAPES:")
    print("\tfeat_factor:", feat_factor.shape)
    print("\tinst_factor:", inst_factor.shape)
    print("\tfeatures:", features.shape)
    print("\tinstances:", instances.shape)
    print("\tpathways:", pathways.shape)

    return feat_factor, inst_factor, features, instances, pathways



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
    true_features, true_instances, true_pwys = load_factor_hdf(args.truth_hdf)

    pred_feat_factor, pred_inst_factor,\
    pred_features, pred_instances, pred_pwys = load_factor_hdf(args.inferred_hdf,
                                                    inst_key="instances",
                                                    feat_key="features")

    score_dict = score_inferred_factors(true_feat_factor, true_inst_factor,
                                        true_features, true_instances, true_pwys,
                                        pred_feat_factor, pred_inst_factor,
                                        pred_features, pred_instances, pred_pwys)

    with open(args.score_json, "w") as f:
        json.dump(score_dict, f)


