# observe_patterns.py
# (c) 2020 David Merrell
# 
# The idea is to generate a list of indices
# telling us which data are observed -- the
# rest are unobserved.

import argparse
import json
import numpy as np
import pandas as pd

def create_tcga_idx(args):

    # We have measurements of the  
    # DNA (CNV) and MRNA (gene expression)
    # corresponding to each of these genes
    df = pd.read_csv(args.tcga_csv_file)
    measured_genes = set(df.columns[1:])

    with open(args.pathway_file) as f:
        pwys = json.load(f)

    idx = []
    for i, name in enumerate(pwys["entity_names"]):
        sym_typ = name.split("::")
        if (sym_typ[0] in measured_genes and sym_typ[1] in ("DNA","MRNA")):
            idx.append(i)

    return idx


def create_random_idx(args):

    with open(args.pathway_file) as f:
        pwys = json.load(f)

    n_entities = len(pwys["entity_names"])
    n_obs = round(n_entities * args.obs_frac)

    idx = np.random.choice(range(n_entities), n_obs, replace=False)
    idx.sort()

    return idx.tolist()


def create_obs_idx(args):

    # Determine which variables are measured
    if args.method == "TCGA":
        measured_idx = create_tcga_idx(args)
    elif args.method == "random":
        measured_idx = create_random_idx(args)

    # Create a random train/test split of the measured variables
    n_test = round(len(measured_idx) * args.test_frac)
    test_idx = np.random.choice(measured_idx, n_test, replace=False).tolist()
    test_set = set(test_idx)
    train_idx = [idx for idx in measured_idx if idx not in test_set]

    result = {"measured_train": train_idx,
              "measured_test": test_idx
              }

    with open(args.output_file, "w") as f:
        json.dump(result, f)

    return

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Generates a list of indices indicating which data are observed.")
    parser.add_argument("method", type=str,
                         choices=["TCGA", "random"], 
                         help="the method for generating the indices.")
    parser.add_argument("pathway_file", type=str,
                         help="path to a JSON file containing preprocessed pathway information")
    parser.add_argument("output_file", type=str,
                         help="path for the output JSON file") 
    parser.add_argument("--tcga-csv-file", type=str, required=False,
                         help="Path to a CSV file of TCGA data. Required iff method=TCGA")
    parser.add_argument("--obs-frac", type=float, required=False,
                         help="fraction of columns to observe. Must take a value in [0, 1]. Required iff method=random")
    parser.add_argument("--test-frac", type=float, required=False, default=0.0,
                         help="fraction of *observed* columns to hold out as a test set.")

    args = parser.parse_args()
    create_obs_idx(args)


