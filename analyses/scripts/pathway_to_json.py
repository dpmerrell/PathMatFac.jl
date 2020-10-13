"""
pathway_to_json.py
2020-08-05 
David Merrell

Move all of the pathway data into a single JSON format
"""

import pandas as pd
import json
import sys
import os


def read_pathway_file(filename):

    entities = {}
    edges = []

    with open(filename, "r") as f:
        for line in f:
            entries = line.strip().split("\t")

            # This line names an entity
            if len(entries) == 2:
                entities[entries[1]] = {"type":entries[0]}
            # This line gives an edge
            elif len(entries) == 3:
                assert entries[0] in entities
                assert entries[1] in entities
                edges.append(entries)

    pathway_info = {"entities": entities,
                    "edges": edges
                    }

    return pathway_info


def read_all_pathways(names_file, pathway_files):

    names_df = read_names_file(names_file)

    results = {}
    for fname in pathway_files:
        print(fname)
        pathway_name = get_pathway_name(fname, names_df)
        pathway_info = read_pathway_file(fname)
        results[pathway_name] = pathway_info

    return results


def get_pathway_name(filename, names_df):
    idx = int(os.path.basename(filename).split("_")[1])
    return names_df.loc[idx, "pathway_name"]


def read_names_file(filename):

    names_df = pd.read_csv(filename, sep="\t")
    names_df.set_index(["pathway_id"], inplace=True)
    return names_df


if __name__=="__main__":

    names_file = sys.argv[1]
    pathway_files = sys.argv[2:-1]
    out_file = sys.argv[-1]

    results = read_all_pathways(names_file, pathway_files)

    with open(out_file, "w") as f:
        json.dump(results, f)

