
import numpy as np
import argparse
import h5py
import json


def load_patient_hierarchy(dataset_hdf):
    
    with h5py.open(dataset_hdf) as f:
        cancer_types = f["cancer_types"][:]
        patients = f["columns"][:]
    
    hierarchy = {str(ct):[] for ct in np.unique(cancer_types)}
    for pat, ct in zip(patients, cancer_types):
        hierarchy[ct].append(pat) 

    return hierarchy


def split_hierarchy(hierarchy, held_out_ctypes, 
                               held_out_patients)

    train_hierarchy = {ct: [] for ct in hierarchy.keys() if ct not in held_out_ctypes}
    test_hierarchy = {ct: [] for ct in hierarchy.keys() if ct in held_out_ctypes}

    if held_out_patients is not None:
        for ct, pat_ls in hierarchy.items():
            for pat in pat_ls:
                if pat not in held_out_patients:
                    train_hierarchy[ct].append(pat)
                else:
                    test_hierarchy[ct].append(pat) 

    else:
        for ct, pat_ls in hierarchy.items():
            if ct in held_out_ctypes:
                test_hierarchy[ct] = pat_ls
            else:
                train_hierarchy[ct] = pat_ls

    return train_hierarchy, test_hierarchy


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_hdf")
    parser.add_argument("train_set_json")
    parser.add_argument("test_set_json")
    parser.add_argument("--hold-out-ctypes", nargs="+")
    parser.add_argument("--hold-out-patients", required=False, nargs="+")

    args = parser.parse_args()

    hierarchy = load_patient_hierarchy(args.dataset_hdf)    

    train, test = split_hierarchy(hierarchy, args.hold_out_ctypes,
                                             args.hold_out_patients)

    with open(args.train_set_json) as f:
        json.dump(train, f)

    with open(args.test_set_json) as f:
        json.dump(test, f)



