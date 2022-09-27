"""
    Given an HDF5 file of TCGA data, extract its
    samples, barcodes, and features and save them 
    to JSON files.
"""


import script_util as su
import h5py
import json
import sys


def get_sample_ids(in_hdf):
    return list(su.load_hdf(in_hdf, "omic_data/instances", dtype=str))

def get_sample_groups(in_hdf):
    return list(su.load_hdf(in_hdf, "omic_data/instance_groups", dtype=str))

def get_barcodes(in_hdf):
    assays = su.load_hdf(in_hdf, "barcodes/features", dtype=str)
    barcode_arr = su.load_hdf(in_hdf, "barcodes/data", dtype=str)
    result = {assays[i] : list(barcode_arr[i,:]) for i,_ in enumerate(assays)}
    return result

def get_features(in_hdf):
    #features = su.load_hdf(in_hdf, "omic_data/features", dtype=str)
    #split_features = [f.split("_") for f in features]
    #assays = [f[-1] for f in split_features]
    #genes = ["_".join(f[:-1]) for f in split_features]
    genes = list(su.load_hdf(in_hdf, "omic_data/feature_genes", dtype=str))
    assays = list(su.load_hdf(in_hdf, "omic_data/feature_assays", dtype=str))
    return genes, assays


if __name__=="__main__":

    in_hdf = sys.argv[1]
    sample_json = sys.argv[2]
    features_json = sys.argv[3]

    sample_ids = get_sample_ids(in_hdf) 
    sample_groups = get_sample_groups(in_hdf)
    barcodes = get_barcodes(in_hdf) 

    sample_dict = { "sample_ids": sample_ids,
                    "sample_groups": sample_groups,
                    "barcodes": barcodes
                  }
    json.dump(sample_dict, open(sample_json, "w")) 


    feature_genes, feature_assays = get_features(in_hdf)
    feature_dict = {"feature_genes": feature_genes,
                    "feature_assays": feature_assays
                   } 

    json.dump(feature_dict, open(features_json, "w"))


