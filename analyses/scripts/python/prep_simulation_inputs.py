"""
Prepare JSON files that define the layout of a 
simulated dataset. These can then be fed to the
data simulator.

Inputs:
    * 
Outputs:
    * 
"""

from collections import Counter
import numpy as np
import sys
import json


def get_pwy_nodes(pwy):
    nodes = set()
    for edge in pwy:
        nodes.add(edge[0])
        nodes.add(edge[2])
    return nodes

def get_all_pwy_nodes(pwy_ls):
    nodes = set()
    for pwy in pwy_ls:
        nodes |= get_pwy_nodes(pwy)
    return nodes

def filter_features(in_features, kept_pwy_dict, n_features, out_features_json):

    in_genes = in_features["feature_genes"]
    in_assays = in_features["feature_assays"]

    # For each pathway, find the intersection between
    # that pathway's genes and the genes in the data 
    data_genes = set(in_genes)
    pwy_node_set = get_all_pwy_nodes(kept_pwy_dict["pathways"]) 
    pwy_data_genes = pwy_node_set.intersection(data_genes)
    nonpwy_data_gene_set = data_genes.difference(pwy_data_genes)

    # Now find all of the feature indices corresponding
    # to those genes
    pwy_data_idx = [idx for idx, g in enumerate(in_genes) if g in pwy_data_genes]

    # If we have too many features, remove them at random
    diff = n_features - len(pwy_data_idx)
    if diff < 0:
        to_remove = set(np.random.choice(pwy_data_idx, -diff, replace=False))
        out_idx = [idx for idx in pwy_data_idx if idx not in to_remove]

    # If we have too few features, add them at random
    # (but group them by gene.)
    elif diff > 0:
        # Create a map from non-pathway genes to random numbers
        nonpwy_data_genes = list(nonpwy_data_gene_set)
        gene_to_rand = dict(zip(nonwpwy_data_genes, np.random.permutation(len(nonpwy_data_genes))))
        # Apply the map to the in_genes, and then sort by 
        # the random numbers. Produces a random permutation of 
        # the in_genes, grouped by gene.
        rand_gene_labels = list(map(lambda x: gene_to_rand[x], in_genes))
        srt_idx = np.argsort(rand_gene_labels)
        
        # Get the indices of the non-pathway features in the data
        nonpwy_data_idx = [idx for idx, g in enumerate(in_genes) if g in nonpwy_data_gene_set]
        srt_nonpwy_data_idx = [nonpwy_data_idx[idx] for idx in srt_idx]
        out_idx = pwy_data_idx + srt_nonpwy_data_idx[:diff]
    else: # For the unlikely case that we got it just right:
        out_idx = pwy_data_idx
        
    # Finally, collect the genes and assays belonging
    # to the selected indices. Save to JSON.
    out_genes = [in_genes[i] for i in out_idx]
    out_assays = [in_assays[i] for i in out_idx]

    result = {"feature_genes": out_genes,
              "feature_assays": out_assays}

    json.dump(result, open(out_features_json, "w"))


def filter_samples(in_samples, m_samples, out_samples_json):

    # For each sample, get the size of its group 
    group_counts = Counter(in_samples["sample_groups"])
    sample_gp_sizes = list(map(lambda x: group_counts[x], in_samples["sample_groups"]))
 
    # Sort the samples by their group membership: small groups -> large groups 
    srt_idx = np.argsort(sample_gp_sizes)

    # Keep the first `m_samples` of these sorted indices
    kept_idx = srt_idx[:m_samples]

    # Select the corresponding samples
    sample_ids = in_samples["sample_ids"]
    sample_groups = in_samples["sample_groups"]
    sample_barcodes = in_samples["barcodes"]
    
    out_sample_ids = [sample_ids[i] for i in kept_idx]
    out_sample_groups = [sample_groups[i] for i in kept_idx]
    out_sample_barcodes = {k : [v[i] for i in kept_idx] for k, v in sample_barcodes.items()}

    # Output to JSON
    result = {"sample_ids": out_sample_ids,
              "sample_groups": out_sample_groups,
              "barcodes": out_sample_barcodes}

    json.dump(result, open(out_samples_json, "w"))



def filter_pwys(in_pwy_dict, k_pwys, out_pwy_json):
    
    # Choose the k bottom-ranked pathways
    n_all_pwys = len(in_pwy_dict["names"])
    kept_pwy_idx = list(range(n_all_pwys-k_pwys, n_all_pwys))
    
    # Save to JSON
    kept_pwys = {"names": [in_pwy_dict["names"][idx] for idx in kept_pwy_idx],
                 "pathways": [in_pwy_dict["pathways"][idx] for idx in kept_pwy_idx]
                }
    
    json.dump(kept_pwys, open(out_pwy_json, "w"))

    return kept_pwys


if __name__ == "__main__":

    # Get arguments       
    in_pwy_json = sys.argv[1]
    in_samples_json = sys.argv[2]
    in_features_json = sys.argv[3]

    m_samples = int(sys.argv[4])
    k_pwys = int(sys.argv[5])
    n_features = int(sys.argv[6])

    out_pwy_json = sys.argv[7]
    out_samples_json = sys.argv[8] 
    out_features_json = sys.argv[9]

    # We use K random pathways
    in_pwy_dict = json.load(open(in_pwy_json, "r"))
    kept_pwys = filter_pwys(in_pwy_dict, k_pwys, out_pwy_json)
    
    # We use M samples, grouped by cancer type
    in_samples_dict = json.load(open(in_samples_json, "r"))
    filter_samples(in_samples_dict, m_samples, out_samples_json)

    # We use N features, chosen in such a way that 
    # an appropriate fraction of them belong to our chosen
    # pathways
    in_features_dict = json.load(open(in_features_json, "r"))
    filter_features(in_features_dict, kept_pwys, n_features, out_features_json)


