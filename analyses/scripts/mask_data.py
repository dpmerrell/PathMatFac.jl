
import script_util as su
import numpy as np
import argparse
import h5py
import json



def create_mask(all_instances, all_features, mask_params): 

    m = len(all_instances)
    n = len(all_features)
    mxn = m * n 

    print("MASK PARAMS:")
    print(mask_params)
    
    if "droprand" in mask_params.keys():
    
        mask_fraction = float(mask_params["droprand"][0])
        n_mask = int(mask_fraction * mxn)
        mask_idx = np.random.choice(mxn, n_mask, replace=False)
    
        mask_coords = [(idx // n, idx % n) for idx in mask_idx]

        data_mask = [(all_instances[i], all_features[j]) for (i,j) in mask_coords]

    else:
        raise ValueError

    return data_mask


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_hdf")
    parser.add_argument("mask_json")
    parser.add_argument("--mask-str")

    args = parser.parse_args()

    all_patients = su.load_instances(args.dataset_hdf)
    all_features = su.load_features(args.dataset_hdf)    

    mask_params = su.parse_arg_str(args.mask_str)

    data_mask = create_mask(all_patients, all_features, 
                            mask_params)

    with open(args.mask_json, "w") as f:
        json.dump(data_mask, f)



