
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
    print("MUT FEATURES:", len(mut_features))

    omic_data = su.load_data(omic_hdf).transpose()
    print("OMIC DATA:", omic_data.shape)
    omic_data = omic_data[:,mut_idx]

    print("ROW SUMS:")
    print(omic_data.sum(axis=1))

    return omic_data, mut_features


def to_txt_file(mutation_data, instances, features, output_txt):

    print(mutation_data)
    idx = np.argwhere(mutation_data != 0.0)
    print(idx)
    print(idx.shape)
    genes = [feat.split("_")[0] for feat in features]

    lines = ["{}\t{}\n".format(instances[idx[i,0]],genes[idx[i,1]]) for i in range(idx.shape[0])]

    with open(output_txt, "w") as f:    
        f.writelines(lines)

    return


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("omic_hdf", help="HDF5 file containing mutation data (among other kinds of data)")
    parser.add_argument("output_txt", help="Output text file, formatted to pyNBS's requirements")

    args = parser.parse_args()

    mutation_data, mut_features = get_mutation_data(args.omic_hdf)
    instances = su.load_instances(args.omic_hdf)
    
    to_txt_file(mutation_data, instances, mut_features, args.output_txt)


    
