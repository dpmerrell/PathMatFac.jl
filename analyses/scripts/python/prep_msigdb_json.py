"""
    Input:
        * JSONs of MSigDB gene sets (from http://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp)
    Outputs:
        * A JSON of "pathways"
"""

import json
import sys
import numpy as np


# Create a trivial "pathway" that just has self-edges
def geneset_to_pwy(geneset):
    return [[g,g,1] for g in geneset]


if __name__=="__main__":

    input_jsons = sys.argv[1:-1]
    output_json = sys.argv[-1]

    print("input_jsons: ", input_jsons)
    print("output_json: ", output_json)

    all_genesets = []
    all_names = []
    for input_json in input_jsons:
        with open(input_json, "r") as f:
            contents = json.load(f)

            gs_names = list(contents.keys())
            all_names += gs_names

            genesets = [contents[k]["geneSymbols"] for k in gs_names]
            all_genesets += genesets


    all_pwys = [geneset_to_pwy(gs) for gs in all_genesets]

    results = {"pathways": all_pwys,
               "names": all_names
              }

    with open(output_json, "w") as f:
        json.dump(results, f)
    


