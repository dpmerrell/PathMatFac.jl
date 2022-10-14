
import sys
import h5py
import numpy as np


def process_missingness_opts(missingness_opts):
    pairs = [opt.split("=") for opt in missingness_opts]
    return {pair[0]: float(pair[1]) for pair in pairs}


def apply_missingness(in_hdf, out_hdf, fracs={"mrnaseq": 0.0,
                                              "cna": 0.0,
                                              "mutation": 0.0,
                                              "methylation": 0.0,
                                              "rppa": 0.0}, each=None)
    # Handle kwargs
    if each is not None:
        for k in fracs.keys():
            fracs[k] = each

    # Load the barcodes
    barcodes = load_hdf(in_hdf, "barcodes/data", dtype=str) # (assays) x (samples) 
    barcode_instances = load_hdf(in_hdf, "barcodes/instances", dtype=str) # (samples)
    barcode_features = load_hdf(in_hdf, "barcodes/features", dtype=str) # (assays)
    assay_to_idx = {assay:idx for idx, assay in enumerate(barcode_features)}
    A, M = barcodes.shape 

    # We'll use this array to indicate which measurements are being nulled out
    barcode_mask = np.zeros(A,M, dtype=bool)
    
    # For each assay, determine which samples will be nulled out
    for assay in fracs.keys():
        frac = fracs[assay]
        to_remove = round(frac*M)
        assay_idx = assay_to_idx[assay]

        # Determine the unique batches and their sizes.
        unique_batches, batch_counts = np.unique(barcodes[assay_idx,:])
        batch_counts = dict(zip(unique_batches, batch_counts))

        # Step through (randomly ordered) batches, adding samples
        # until we've nulled out the required number 
        randomized_batches = np.random.permutation(unique_batches)
        batch_idx = 0
        while to_remove > 0:
            # Identify the samples belonging to this batch
            batch = randomized_batches[batch_idx]
            thisbatch_indices = np.argwhere(barcodes[assay_idx,:] == batch)

            # Flag the appropriate number of samples to null out
            thisbatch_toremove = min(to_remove, batch_counts[batch])
            toremove_idx = np.random.choice(thisbatch_indices, thisbatch_toremove, replace=False)
            barcode_mask[assay_idx, toremove_idx] = True

    # Load the column assays; build a map from assay to columns
    data_assays = load_hdf(in_hdf, "omic_data/feature_assays", dtype=str)
    data_genes = load_hdf(in_hdf, "omic_data/feature_genes", dtype=str)
    assay_to_columns = {assay:np.argwhere(data_assays == assay) for assay in barcode_features}
   
    data_instances = load_hdf(in_hdf, "omic_data/instances", dtype=str)
    data_groups = load_hdf(in_hdf, "omic_data/instance_groups", dtype=str)
    # Load the rest of the data. Time to mask it!
    data = load_hdf(in_hdf, "omic_data/data")
    for assay, columns in assay_to_columns.items():
        mask_rows = barcode_mask[assay_to_idx[assay],:]
        for col in columns:
            data[col, mask_rows] = np.nan

    # mask the barcodes, too
    masked_barcodes = np.where(barcode_mask, "", barcodes) 

    with h5py.File(out_hdf, "w") as fout:
        fout["barcodes/data"] = masked_barcodes
        fout["barcodes/instances"] = barcode_instances
        fout["barcodes/features"] = barcode_features

        fout["omic_data/data"] = data
        fout["omic_data/feature_assays"] = data_assays
        fout["omic_data/feature_genes"] = data_genes
        fout["omic_data/instances"] = data_instances
        fout["omic_data/instance_groups"] = data_groups
 
    return


if __name__=="__main__":

    args = sys.argv
    in_hdf = args[1]
    out_hdf = args[2]
    
    missingness_opts = args[3:]

    missingness_dict = process_missingness_opts(missingness_opts)

    apply_missingness(in_hdf, out_hdf, **missingness_dict)

