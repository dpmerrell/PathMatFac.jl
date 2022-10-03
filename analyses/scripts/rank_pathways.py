"""
    Inputs:
        * A JSON of pathways
        * A JSON of features (i.e., genes)
    Outputs:
        * A JSON of pathways, *ranked* by their
          coverage of the genes. I.e., we run a greedy
          set coverage algorithm which produces an
          ordering on the pathways; ties are broken
          by pathway size.
"""

import json
import sys
import numpy as np


def pwy_to_geneset(pwy, all_genes):
    
    geneset = set() 
    for edge in pwy:
        for protein in (edge[0], edge[-1]):
            if protein in all_genes:
                geneset.add(protein)

    return geneset


def rank_genesets(genesets, top_k=None):

    ranked = []
    scores = []

    covered = set()

    if top_k is None:
        top_k = len(genesets)

    i = 0
    while i < top_k:

        # Score by newly-covered genes;
        # but break ties with total number of covered genes
        max_score = (-1,-1)
        max_k = ""
        for k, gs in genesets.items():
            score = (len(gs.difference(covered)), len(gs))
            if score > max_score:
                max_score = score
                max_k = k

        print(max_score, "\t", max_k)
        ranked.append(max_k)
        scores.append(max_score)
        covered |= genesets[max_k]
        genesets.pop(max_k)

        i += 1

    return ranked, scores
             

def main():

    input_json = sys.argv[1]
    features_json = sys.argv[2]
    output_json = sys.argv[3]

    with open(input_json, "r") as f:
        all_pwys = json.load(f)
        all_pwys = dict(zip(all_pwys["names"], all_pwys["pathways"]))

    with open(features_json, "r") as f:
        features = json.load(f)

    all_genes = set(features["feature_genes"])

    print(len(all_pwys))

    #all_genes = get_all_pwy_genes(all_pwys, all_genes)

    print("making gene sets...")
    genesets = {k: pwy_to_geneset(pwy, all_genes) for k, pwy in all_pwys.items()}

    print("ranking gene sets...")
    ranked, scores = rank_genesets(genesets)

    results = {"pathways": [all_pwys[k] for k in ranked],
               "names": ranked,
               "pathway_scores": scores}

    with open(output_json, "w") as f:
        json.dump(results, f)
    

if __name__=="__main__":

    main()


