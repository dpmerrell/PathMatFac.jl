
import script_util as su
import pandas as pd
import numpy as np
import argparse
import json
import h5py


def get_mutation_data(omic_hdf):

    features = su.load_features(omic_hdf).astype(str)
    mut_idx = [idx for idx, feat in enumerate(features) if feat.endswith("mutation")]
    mut_features = [feat for idx, feat in enumerate(features) if feat.endswith("mutation")]

    omic_data = su.load_data(omic_hdf)
    omic_data = omic_data[:,mut_idx]

    print("OMIC DATA:", omic_data.shape)
    print("MUT FEATURES:", len(mut_features))

    return omic_data, mut_features


def split_and_mask(omic_data, instances, features, split_d, mask_ls):

    train_idx = split_d["train"]

    omic_data = omic_data[train_idx,:]
    print("TRAINING DATA", omic_data.shape)
    instances = instances[train_idx]
    print("TRAINING INSTANCES:", instances.shape)
    train_inst_set = set(instances)

    train_mask = [coord for coord in mask_ls if coord[0] in train_inst_set]
    train_mask = [coord for coord in train_mask if coord[1].split("_")[-1] == "mutation"]
    print("TRAIN MASK: ", len(train_mask))

    omic_data = su.apply_mask(omic_data, instances, features, train_mask) 

    return omic_data, instances, features



def to_txt_file(mutation_data, instances, features, output_txt):

    idx = np.argwhere(mutation_data == 1)
    genes = [feat.split("_")[0] for feat in features]

    lines = ["{}\t{}\n".format(instances[idx[i,0]],genes[idx[i,1]]) for i in range(idx.shape[0])]

    with open(output_txt, "w") as f:    
        f.writelines(lines)

    return


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("omic_hdf", help="HDF5 file containing mutation data (among other kinds of data)")
    parser.add_argument("split_json", help="JSON file specifying indices of train and test sets")
    parser.add_argument("mask_json", help="JSON file containing coordinates of held-out data")
    parser.add_argument("output_txt", help="Output text file, formatted to pyNBS's requirements")

    args = parser.parse_args()

    mutation_data, mut_features = get_mutation_data(args.omic_hdf)
    instances = su.load_instances(args.omic_hdf)
    
    split_d = json.load(open(args.split_json, "r"))
    mask_ls = json.load(open(args.mask_json, "r"))

    mutation_data, instances, features = split_and_mask(mutation_data,
                                                        instances, mut_features, 
                                                        split_d, mask_ls)

    to_txt_file(mutation_data, instances, features, args.output_txt)


    
