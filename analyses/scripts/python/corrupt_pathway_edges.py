"""
randomize_pathways.py

A script that receives a set of pathways
and returns a "randomized" version of each one.
Each pathway is randomized such that nodes' degrees
are preserved (and hence degree distribution is preserved).
"""

from collections import defaultdict
import numpy as np
import json
import sys


def create_node_encoding(edgelist):
    nodes = set()
    for edge in edgelist:
        if edge[0] != edge[2]:
            nodes.add(edge[0])
            nodes.add(edge[2])
    nodes = sorted(list(nodes))
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    return nodes, node_to_idx


def prepare_edges(edgelist):

    pos_edgelist = [tuple(sorted([edge[0], edge[2]])) for edge in edgelist if edge[1][-1] == ">"]
    pos_edgelist = sorted(list(set(pos_edgelist)))
    neg_edgelist = [tuple(sorted([edge[0], edge[2]])) for edge in edgelist if edge[1][-1] == "|"]
    neg_edgelist = sorted(list(set(neg_edgelist)))
    return pos_edgelist, neg_edgelist


def compute_degrees(edgelist):

    degrees = defaultdict(lambda : 0)
    for edge in edgelist:
        degrees[edge[0]] += 1
        degrees[edge[1]] += 1

    degree_list = [0]*len(degrees)
    for k,v in degrees.items():
        degree_list[k] = v

    return degree_list


def degree_preserving_random_graph(degrees, verbose=False):

    V = len(degrees)
    deg = np.array(degrees, dtype=int)
    orig_deg = np.copy(deg)
    adjacency = np.zeros((V,V), dtype=bool)
    np.fill_diagonal(adjacency, True)

    edgelist = []

    while not np.all(deg == 0):
       
        # Select first node 
        i = np.argmax(deg)
        
        # Select second node at random from non-neighbors
        non_neighbors = np.ravel(np.argwhere(adjacency[:,i] == False))
        weight_vec = (deg[non_neighbors] > 0)
        if np.all(weight_vec == False):
            if verbose:
                print("WARNING: graph randomization: forced to violate node degree preservation")
            weight_vec = orig_deg[non_neighbors]

        weight_vec = weight_vec.astype(float)
        weight_vec /= np.sum(weight_vec)
        j = np.random.choice(non_neighbors, p=weight_vec)

        # Update degree vector and adjacency matrix
        deg[i] = max(0, deg[i] - 1)
        deg[j] = max(0, deg[j] - 1)
        adjacency[i,j] = True
        adjacency[j,i] = True

        # Add edge to edgelist
        i_srt = min(i,j)
        j_srt = max(i,j)
        edgelist.append((i_srt, j_srt))
        
    return edgelist


def randomize_pathway(edgelist):

    nodes, node_to_idx = create_node_encoding(edgelist)
    encoded_edgelist = [(node_to_idx[edge[0]], edge[1], node_to_idx[edge[2]]) for edge in edgelist]
    
    pos_edgelist, neg_edgelist = prepare_edges(encoded_edgelist)
    pos_degrees = compute_degrees(pos_edgelist)
    randomized_pos_edges = degree_preserving_random_graph(pos_degrees)
    decoded_pathway = [[nodes[i], "a>", nodes[j]] for (i,j) in randomized_pos_edges]

    neg_degrees = compute_degrees(neg_edgelist)
    randomized_neg_edges = degree_preserving_random_graph(neg_degrees)
    decoded_pathway += [[nodes[i], "a|", nodes[j]] for (i,j) in randomized_neg_edges]

    return decoded_pathway 


def randomize_all_pathways(pathway_dict):

    randomized_pwys = []
    for k, pwy in enumerate(pathway_dict["pathways"]):
        randomized_pwys.append(randomize_pathway(pwy))
        print(k, pathway_dict["names"][k])

    return {"pathways": randomized_pwys,
            "names": pathway_dict["names"]}


if __name__=="__main__":

    args = sys.argv
    pwy_json = args[1]
    out_json = args[2]

    input_pwys = json.load(open(pwy_json, "r"))
    
    randomized_pwys = randomize_all_pathways(input_pwys)

    json.dump(randomized_pwys, open(out_json, "w"))


