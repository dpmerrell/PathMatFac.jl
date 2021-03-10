
import h5py
import numpy as np
import argparse 


def load_factor_hdf(factor_hdf, feat_key="index", inst_key="columns"):

    with h5py.File(factor_hdf, "r") as f:
        feat_factor = f["feature_factor"][:,:]
        inst_factor = f["instance_factor"][:,:]
        features = f[feat_key][:]
        instances = f[inst_key][:]

    print("SHAPES:")
    print("\tfeat_factor:", feat_factor.shape)
    print("\tinst_factor:", inst_factor.shape)
    print("\tfeatures:", features.shape)
    print("\teatures:", features.shape)

    return feat_factor, inst_factor, features, patients



def score_inferred_factors(true_feat_factor, true_inst_factor,
                           true_features, true_instances,
                           pred_feat_factor, pred_inst_factor,
                           pred_features, pred_instances):
    print("WE GOT THIS FAR")
    pass



if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("truth_hdf", help="path to HDF file containing true factors")
    parser.add_argument("inferred_hdf", help="path to HDF file containing inferred factors")
    parser.add_argument("score_json", help="path to output JSON")

    args = parser.parse_args()

    true_feat_factor, true_inst_factor,\
    true_features, true_instances = load_factor_hdf(args.truth_hdf)

    pred_feat_factor, pred_inst_factor,\
    pred_features, pred_instances = load_factor_hdf(args.inferred_hdf,
                                                    inst_key="instances",
                                                    feat_key="features")

    score_dict = score_inferred_factors(true_feat_factor, true_inst_factor,
                                        true_features, true_instances,
                                        pred_pat_factor, pred_inst_factor,
                                        pred_features, pred_instances)

    with open(args.score_json, "w") as f:
        json.dump(score_dict, f)


