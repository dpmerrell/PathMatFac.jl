
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


def load_name_map(names_file):

    name_df = pd.read_csv(names_file, sep="\t")
    
    name_map = {row["pathway_id"]:row["pathway_name"] for _, row in name_df.iterrows()}
    return name_map


def extract_id(file_path):
    basename = os.path.basename(file_path)
    return int(basename.split(".")[0].split("_")[1])


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sif-files", help="One or more pathway SIF files to include in the JSON", nargs="+")
    parser.add_argument("--names-file", help="Text file that maps SIF files to pathway names")
    parser.add_argument("--out-json", help="Output JSON file")

    args = parser.parse_args()

    pathways = [sif_to_edgelist(sif) for sif in args.sif_files]
    
    id_to_name = load_name_map(args.names_file)

    result = { "pathways": pathways,
               "names": [id_to_name[extract_id(pth)] for pth in args.sif_files]
             }
    
    with open(args.out_json, "w") as fout:
        json.dump(result, fout)


