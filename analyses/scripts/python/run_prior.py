
import numpy as np
import h5py
import json
import sys

def pwy_to_genes(pwy, all_gene_set):

    gene_ls = []

    for edge in pwy:
        if edge[0] in all_gene_set:
            gene_ls.append(edge[0])
        if edge[2] in all_gene_set:
            gene_ls.append(edge[2])

    return list(set(gene_ls))


def build_gene_to_idx(all_genes):
    
    gene_to_idx = {g: [] for g in set(all_genes)}
    for idx, gene in enumerate(all_genes):
        gene_to_idx[gene].append(idx)

    return gene_to_idx


def generate_Y(all_pwys, all_genes):

    K = len(all_pwys)
    N = len(all_genes)
    prior_Y = np.zeros((N,K), dtype=np.float32)
    
    gene_to_idx = build_gene_to_idx(all_genes)
    all_gene_set = set(all_genes)

    for k, pwy in enumerate(all_pwys):
        pwy_genes = pwy_to_genes(pwy, all_gene_set)
        for gene in pwy_genes:
            idx_ls = gene_to_idx[gene]
            for idx in idx_ls:
                prior_Y[idx, k] = 1        

    return prior_Y


def main():

    args = sys.argv
    pwy_json = args[1]
    feature_json = args[2]
    pred_hdf = args[3]

    all_pwys = None
    with open(pwy_json, "r") as f:
        pathway_dict = json.load(f)
        all_pwys = pathway_dict["pathways"]
 
    all_genes = None
    with open(feature_json, "r") as f:
        feature_dict = json.load(f)
        all_genes = feature_dict["feature_genes"]

    prior_Y = generate_Y(all_pwys, all_genes)

    with h5py.File(pred_hdf, "w") as f:
        f.create_dataset("/Y", prior_Y.shape)
        f["/Y"][:,:] = prior_Y

    

if __name__=="__main__":

    main()
