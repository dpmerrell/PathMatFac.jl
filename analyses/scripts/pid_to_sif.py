"""
pid_to_sif.py
2021-02
David Merrell

Convert the NCI PID files into a 
standardized SIF format
"""

import pandas as pd
import sys
import os


entity_type_map = {"protein": "p",
                   "complex": "c",
                   "abstract": "b",
                   "chemical": "h",
                   "family": "f"
                  }


def read_pid_file(filename):

    entity_types = {}
    edges = []

    with open(filename, "r") as f:
        for line in f:
            try:
                entries = line.strip().split("\t")

                # This line names an entity
                if len(entries) == 2:
                    entity_types[entries[1]] = entity_type_map[entries[0]]
                # This line gives an edge
                elif len(entries) == 3:
                    assert entries[0] in entity_types.keys()
                    assert entries[1] in entity_types.keys()
                    edges.append(entries)
            except:
                print("ERROR: ", line)
                raise ValueError

    pathway_info = {"entity_types": entity_types,
                    "edges": edges
                    }

    return pathway_info


def standardize_edge(edge, entity_types):
    u = edge[0]
    v = edge[1]
    tag = edge[2]
   
    tag_sign = tag[-1] 

    tag_kind = tag[:-1]
    tag_target = ""
    if tag_kind in ('-a', '-ap', 'component', 'member'):
        tag_target = "a"
    elif tag_kind == "-t":
        tag_target = "t"
    else:
        print(tag)
        raise ValueError

    edge = [u, 
            "{}{}{}{}".format(entity_types[u], 
                              entity_types[v], 
                              tag_target, tag_sign), 
            v]

    return edge


def translate_to_sif(pathway_dict): 

    result_ls = []

    for edge in pathway_dict["edges"]:
        result_ls.append(standardize_edge(edge, pathway_dict["entity_types"]))

    result_df = pd.DataFrame(result_ls)

    return result_df


if __name__=="__main__":

    in_pid_file = sys.argv[1]
    out_sif_file = sys.argv[2]

    pathway_dict = read_pid_file(in_pid_file)

    sif_df = translate_to_sif(pathway_dict)

    sif_df.to_csv(out_sif_file, sep="\t", index=False, header=False)


