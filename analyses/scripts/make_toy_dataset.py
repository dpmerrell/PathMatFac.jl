"""
Generate metadata for a toy dataset.
This can then be used to simulate the toy dataset.
"""


import sys
import json

def filter_pwys(in_pwy_json, kept_pwys, out_pwy_json):

    in_pwys = json.load(open(in_pwy_json, "r"))
    out_names = [in_pwys["names"][i] for i in kept_pwys]
    out_pwys = [in_pwys["pathways"][i] for i in kept_pwys]
    result = {"names": out_names,
              "pathways": out_pwys}
    json.dump(result, open(out_pwy_json, "w"))


def filter_samples(in_samples_json, kept_ctypes, out_samples_json):

    ctype_set = set(kept_ctypes)
    in_samples = json.load(open(in_samples_json, "r"))
    
    sample_ctypes = in_samples["sample_groups"]
    sample_ids = in_samples["sample_ids"]
    sample_barcodes = in_samples["barcodes"]
    
    out_sample_idx = [i for i, g in enumerate(sample_ctypes) if g in ctype_set]
    out_sample_ids = [sample_ids[i] for i in out_sample_idx]
    out_sample_groups = [sample_ctypes[i] for i in out_sample_idx]
    out_sample_barcodes = {k : [v[i] for i in out_sample_idx] for k, v in sample_barcodes.items()}

    result = {"sample_ids": out_sample_ids,
              "sample_groups": out_sample_groups,
              "barcodes": out_sample_barcodes}

    json.dump(result, open(out_samples_json, "w"))


def filter_features(in_features_json, kept_genes, out_features_json):

    gene_set = set(kept_genes)
    
    in_features = json.load(open(in_features_json, "r"))
    in_genes = in_features["feature_genes"]
    in_assays = in_features["feature_assays"]

    out_idx = [i for i, g in enumerate(in_genes) if g in gene_set]
    out_genes = [in_genes[i] for i in out_idx]
    out_assays = [in_assays[i] for i in out_idx]

    result = {"feature_genes": out_genes,
              "feature_assays": out_assays}

    json.dump(result, open(out_features_json, "w"))


if __name__ == "__main__":

    # We use 5 small pathways
    kept_pwys = list(range(295, 300))

    # These genes were chosen from those pathways
    kept_genes = ["FLT1", "VEGFA", 
                  "APOB", "LSR", 
                  "ARL3", "CYS1",
                  "CD9", "IZUMO1",
                  "CHEBI:16852", "TYR"]

    # We use the two cancer types with fewest samples
    kept_ctypes = ["CHOL", "DLBC"]

    # Get arguments       
    in_pwy_json = sys.argv[1]
    in_samples_json = sys.argv[2]
    in_features_json = sys.argv[3]

    out_pwy_json = sys.argv[4]
    out_samples_json = sys.argv[5] 
    out_features_json = sys.argv[6]


    filter_pwys(in_pwy_json, kept_pwys, out_pwy_json)
    filter_samples(in_samples_json, kept_ctypes, out_samples_json)
    filter_features(in_features_json, kept_genes, out_features_json)

