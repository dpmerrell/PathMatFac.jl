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



def update_graph(pwy, all_nodes, node_add, node_remove):

    # Get the pathway nodes and unique edges
    pwy_nodes = set()
    unique_edges = set()
    for edge in pwy:
        pwy_nodes.add(edge[0])
        pwy_nodes.add(edge[2])
        if edge[0] <= edge[2]:
            unique_edges.add((edge[0],edge[2]))
        else:
            unique_edges.add((edge[2],edge[0]))


    pwy_node_ls = list(pwy_nodes)
    
    # Figure out how many we're adding and removing
    n_add = int(np.ceil(node_add * len(pwy_nodes)))
    n_remove = int(np.ceil(node_remove * len(pwy_nodes)))

    # Remove nodes
    to_remove = set(np.random.choice(pwy_node_ls, n_remove, replace=False))
    updated_pwy = [edge for edge in pwy if (edge[0] not in to_remove) and (edge[2] not in to_remove)]
    for rm_node in to_remove:
        pwy_nodes.remove(rm_node)
    pwy_node_ls = list(pwy_nodes)

    # Compute the degree of each node
    degrees = defaultdict(lambda : 0)
    for edge in unique_edges:
        degrees[edge[0]] += 1
        degrees[edge[1]] += 1
    degree_vec = np.array([degrees[n] for n in pwy_node_ls])

    # Choose nodes to add
    complement = all_nodes.difference(pwy_nodes)
    nodes_to_add = set(np.random.choice(list(complement), n_add, replace=False))

    # For each new node, add edges to it
    for new_node in nodes_to_add:
        n_neighbors = np.random.choice(degree_vec)
        neighbors = np.random.choice(pwy_node_ls, n_neighbors, replace=False, p=degree_vec/np.sum(degree_vec))
        for neighbor in neighbors:
            updated_pwy.append([neighbor, "a>", new_node])
        pwy_node_ls.append(new_node)

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
    k_add = int(args[3])
    k_remove = int(args[4])
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

