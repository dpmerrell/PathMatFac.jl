
import numpy as np
import argparse
import h5py
import json


def load_patients_ctypes(dataset_hdf):
    
    with h5py.File(dataset_hdf, "r") as f:
        cancer_types = f["cancer_types"][:]
        patients = f["columns"][:]

    return patients.astype(str), cancer_types.astype(str)


def split_patients(patients, ctypes, held_out_ctypes): 

    ho_ct_set = set(held_out_ctypes)

    test_cols = [i+1 for i, ct in enumerate(ctypes) if ct in ho_ct_set]
    train_cols = [i+1 for i, ct in enumerate(ctypes) if ct not in ho_ct_set]

    return train_cols, test_cols 


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_hdf")
    parser.add_argument("split_json")
    parser.add_argument("--hold-out-ctypes", nargs="+")

    args = parser.parse_args()

    patients, ctypes = load_patients_ctypes(args.dataset_hdf)    

    train, test = split_patients(patients, ctypes, 
                                 args.hold_out_ctypes)

    result = {"train": train,
              "test": test,
             }

    with open(args.split_json, "w") as f:
        json.dump(result, f)



