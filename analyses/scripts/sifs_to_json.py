
import pandas as pd
import argparse
import json
import os

def sif_to_edgelist(sif_path):

    edgelist = []
    df = pd.read_csv(sif_path, sep="\t", header=None)
    
    for i, row in df.iterrows():
        edgelist.append([row[0], row[1], row[2]])

    return edgelist
        

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sif-files", help="One or more pathway SIF files to include in the JSON", nargs="+")
    parser.add_argument("--out-json", help="Output JSON file")

    args = parser.parse_args()

    pathways = [sif_to_edgelist(sif) for sif in args.sif_files]
    result = { "pathways": pathways,
               "names": [os.path.basename(pth) for pth in args.sif_files]
             }
    
    with open(args.out_json, "w") as fout:
        json.dump(result, fout)


