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


def read_pid_file(filename):

    edges = []

    with open(filename, "r") as f:
        for line in f:
            try:
                entries = line.strip().split("\t")

                if len(entries) == 3:
                    u = entries[0].upper()
                    v = entries[1].upper()
                    edges.append([u, v, entries[2]])
            except:
                print("ERROR: ", line)
                raise ValueError

    return edges 


def standardize_edge(edge): 
    u = edge[0].upper()
    v = edge[1].upper()
    tag = edge[2]
   
    tag_sign = tag[-1] 
    tag_kind = tag[:-1]

    tag_target = "a"
    if tag_kind == "-t":
        tag_target = "t"
    
    edge = [u, "{}{}".format(tag_target, tag_sign), v]

    return edge


def translate_to_sif(edges): 

    result_ls = [standardize_edge(edge) for edge in edges]

    result_df = pd.DataFrame(result_ls)

    return result_df


if __name__=="__main__":

    in_pid_files = sys.argv[1:-1]
    out_sif_dir = sys.argv[-1]

    for in_pid_file in in_pid_files:

        pathway_edges = read_pid_file(in_pid_file)
        sif_df = translate_to_sif(pathway_edges)
        
        basename = os.path.basename(in_pid_file)
        head = basename.split(".")[0]
        out_path = os.path.join(out_sif_dir, head+".sif")
        print(out_path)
        sif_df.to_csv(out_path, sep="\t", index=False, header=False)




