
import json
import argparse


"""
Convert a pathway into a list of gene IDs.
"""
def pwy_to_geneset(pwy):

    gene_set = set([])

    for edge in pwy:
        u = edge[0]
        u_type = edge[1][0]
        if u_type == "p":
            gene_set.add(u)
        
        v = edge[2]
        v_type = edge[1][1]
        if v_type == "p":
            gene_set.add(v)

    return list(gene_set)



if __name__=="__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("input_json")
    parser.add_argument("output_json")

    args = parser.parse_args()

    with open(args.input_json, "r") as f:
        pwy_dict = json.load(f)
        print("PWY_DICT: ", pwy_dict)

    genesets = [pwy_to_geneset(pwy) for pwy in pwy_dict["pathways"]]
    
    result = { "genesets": genesets,
               "names": pwy_dict["names"]
             }

    with open(args.output_json, "w") as f:
        json.dump(result, f)
   

