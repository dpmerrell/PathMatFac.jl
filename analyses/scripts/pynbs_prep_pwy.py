import script_util as su
import pandas as pd
import argparse
import json
import h5py


def graph_contraction(ugraph, rm_set):

    for k, rm_node in enumerate(rm_set):
        print(k, "removed;", rm_node)

        neighbors = list(ugraph[rm_node])
        for i, n1 in enumerate(neighbors):
            #print("\t", n1)
            ugraph[n1].remove(rm_node)
            for n2 in neighbors[:i-1]:
                add_edge(ugraph, n1, n2)

        ugraph.pop(rm_node)

    return ugraph


def find_all_nonproteins(ugraph):
    
    nonprot = set([node for node in ugraph.keys() if node.split("_")[-1] != "protein"])
    
    return nonprot


def reduce_graph_union(ugraphs):
    result = {}
    for ugraph in ugraphs:
        graph_union(result, ugraph)
    return result


def graph_union(g1, g2):
    for u, neighbors in g2.items():
        for v in neighbors:
            add_edge(g1, u, v)
    return g1


def validate_graph(g):
    for u in g.keys():
        for v in g[u]:
            assert(u in g[v])
    return


def node_tag(node_char):
    if node_char == "p":
        return "protein"
    else:
        return "other"

def to_ugraph(edges):

    ugraph = {}
    for edge in edges:

        ltag = node_tag(edge[1][0])
        rtag = node_tag(edge[1][1])        

        add_edge(ugraph, edge[0]+"_"+ltag, edge[2]+"_"+rtag)

    return ugraph


def add_edge(ugraph, u, v):
    if u == v:
        return
    if u in ugraph.keys():
        ugraph[u].add(v)
    else:
        ugraph[u] = set([v])
    
    if v in ugraph.keys():
        ugraph[v].add(u)
    else:
        ugraph[v] = set([u])

 
def to_txt_file(ugraph, output_txt):

    edge_set = set([])
    for node, neighbors in ugraph.items():
        for neighbor in neighbors:
            if node < neighbor:
                edge_set.add((node, neighbor))
            else:
                edge_set.add((neighbor, node))


    with open(output_txt, "w") as f:
        lines = ["{}\t{}\n".format(edge[0][:-9], edge[1][:-9]) for edge in edge_set]
        f.writelines(lines)

    return


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("pathways_json", help="JSON file containing pathway directed graphs")
    parser.add_argument("output_txt", help="output tab-delimited file containing all undirected edges")

    args = parser.parse_args()

    pathways = json.load(open(args.pathways_json, "r"))
   
    ugraphs = [to_ugraph(pwy) for pwy in pathways["pathways"]]

    for ug in ugraphs:
        validate_graph(ug)

    ugraph = reduce_graph_union(ugraphs)
    validate_graph(ugraph)

    nonprot = find_all_nonproteins(ugraph)

    #print(ugraph)
    #print(nonprot)

    contracted = graph_contraction(ugraph, nonprot)

    to_txt_file(contracted, args.output_txt)

