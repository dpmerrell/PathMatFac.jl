"""
    Inputs:
        * A JSON of true pathways
        * A JSON of all pathways (superset of true pathways)
        * various corruption parameters
    Output:
        * A JSON of corrupted pathways
"""


from collections import defaultdict
import numpy as np
import json
import sys


def get_pwy_nodes(pwy):
    nodes = set()
    for edge in pwy:
        nodes.add(edge[0])
        nodes.add(edge[2])
    return nodes


def get_unq_pwy_edges(pwy):
    unq_edges = set()
    for edge in pwy:
        if edge[0] <= edge[2]:
            unq_edges.add((edge[0],edge[2]))
        else:
            unq_edges.add((edge[2],edge[0]))
    return unq_edges


def get_node_degrees(pwy):
    unique_edges = get_unq_pwy_edges(pwy)
    degrees = defaultdict(lambda : 0)
    for edge in unique_edges:
        degrees[edge[0]] += 1
        degrees[edge[1]] += 1

    return degrees


def update_graph(pwy, all_nodes, node_add, node_remove):

    # Get the pathway nodes and unique edges
    pwy_nodes = get_pwy_nodes(pwy)
    pwy_node_ls = sorted(list(pwy_nodes))
    complement_ls = sorted(list(all_nodes.difference(pwy_nodes)))

    # Compute the set of nodes to remove
    n_remove = int(np.ceil(node_remove * len(pwy_nodes)))
    to_remove = set(np.random.choice(pwy_node_ls, n_remove, replace=False))    

    # Construct updated versions of the pathway and
    # its nodes, with the specified nodes removed 
    updated_pwy = [edge for edge in pwy if (edge[0] not in to_remove) and (edge[2] not in to_remove)]
    updated_node_ls = [node for node in pwy_node_ls if node not in to_remove]
    n_remaining = len(updated_node_ls)

    # Compute the degree of each remaining node
    degree_dict = get_node_degrees(updated_pwy)
    degree_ls = [degree_dict[n] for n in updated_node_ls]
    degree_vec = np.array(degree_ls).astype(float) + 1e-10
    p_vec = degree_vec / np.sum(degree_vec)
    # Compute the set of nodes to add
    n_add = int(np.ceil(node_add * len(pwy_nodes)))
    to_add = set(np.random.choice(complement_ls, n_add, replace=False))

    # For each new node, add edges to it
    for new_node in to_add:
        # Choose the new node's degree at random
        n_neighbors = np.random.choice(degree_ls)
        # Select the new node's neighbors at random from the updated node list, 
        # weighted by their original degree 
        neighbor_idx = np.random.choice(n_remaining, n_neighbors, replace=False, p=p_vec)
        for idx in neighbor_idx:
            updated_pwy.append([updated_node_ls[idx], "a>", new_node])
        updated_node_ls.append(new_node)

    return updated_pwy



def update_graphs(used_pwys, node_add, node_remove):

    all_nodes = set()
    for pwy in used_pwys["pathways"]:
        for edge in pwy:
            all_nodes.add(edge[0])
            all_nodes.add(edge[1])

    return {"names": used_pwys["names"],
            "pathways": [update_graph(pwy, all_nodes, node_add, node_remove) for pwy in used_pwys["pathways"]]}



def update_k(true_pwy_dict, all_pwys_dict, k_add, k_remove, considered=500):
   
    # We may wish to only consider the top K pathways
    # (e.g., the top 500). 
    if considered is None:
        considered = len(all_pwys_dict["names"])
    all_pwys_dict = {"names": all_pwys_dict["names"][:considered],
                     "pathways": all_pwys_dict["pathways"][:considered]}

    true_k = len(true_pwy_dict["pathways"])
    k_add = round(k_add * true_k)
    k_remove = round(k_remove * true_k)

    # Build a map from pathway names to their indices in the full list
    name_to_idx = {name: idx for idx, name in enumerate(all_pwys_dict["names"])}
    
    true_idx = [name_to_idx[name] for name in true_pwy_dict["names"]]

    # Select indices to remove and to add
    to_remove = set(np.random.choice(true_idx, k_remove, replace=False)) 

    complement_idx = set(range(len(all_pwys_dict["names"]))).difference(true_idx) 
    to_add = list(np.random.choice(list(complement_idx), k_add))

    # Build the list of selected indices
    used_idx = [idx for idx in true_idx if idx not in to_remove]
    used_idx += to_add

    # Use the selected indices to construct the pathway set
    used_pwys = {"names": [all_pwys_dict["names"][i] for i in used_idx],
                 "pathways": [all_pwys_dict["pathways"][i] for i in used_idx]
                }

    return used_pwys


def main():

    args = sys.argv
    true_pwy_json = args[1]
    all_pwys_json = args[2]
    k_add = float(args[3])
    k_remove = float(args[4])
    node_add = float(args[5])
    node_remove = float(args[6])
    out_json = args[7]

    true_pwy_dict = json.load(open(true_pwy_json, "r"))
    all_pwys_dict = json.load(open(all_pwys_json, "r"))

    used_pwys = update_k(true_pwy_dict, all_pwys_dict, k_add, k_remove)

    used_pwys = update_graphs(used_pwys, node_add, node_remove)

    json.dump(used_pwys, open(out_json, "w"))



if __name__=="__main__":

    main()

