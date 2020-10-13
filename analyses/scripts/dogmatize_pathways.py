
import json
import sys

def dogmatize_entities(pathway):
    
    dogmatized = [] 

    for k, v in pathway["entities"].items():

        # For proteins, the dogmatized representation
        # becomes four entities: DNA, MRNA, PROT, ACT
        if v["type"] == "protein":
            dogmatized.append(k+"::DNA")
            dogmatized.append(k+"::MRNA")
            dogmatized.append(k+"::PROT")
            dogmatized.append(k+"::ACT")

        else:
            dogmatized.append(k+"::ACT")

    pathway["entities"] = dogmatized

    return pathway


def dogmatize_edge(edge):
    u = edge[0]
    v = edge[1]
    tag = edge[2]
    tag_kind = tag[:-1]
    if tag_kind in ('-a', '-ap', 'component', 'member'):
        edge = [u+"::ACT", v+"::ACT"]
    elif tag_kind == "-t":
        edge = [u+"::ACT", v+"::MRNA"]
    else:
        print(tag)
        raise ValueError

    if tag[-1] == ">":
        edge.append("+")
    elif tag[-1] == "|":
        edge.append("-")

    return edge


def dogmatize_edges(pathway):

    dogmatized = []

    # Add "central dogma" edges
    for ent in pathway["entities"]:
        name_level = ent.split("::")
        name = name_level[0] 
        level = name_level[-1]
        if level == "PROT":
            dogmatized += [[name+"::DNA", name+"::MRNA", "+"],
                           [name+"::MRNA", name+"::PROT", "+"],
                           [name+"::PROT", name+"::ACT", "+"]
                          ]

    # Add the other edges
    for edge in pathway["edges"]:
        dogmatized.append(dogmatize_edge(edge))

    pathway["edges"] = dogmatized

    return pathway


def dogmatize_all_pathways(pathway_data):

    result = {}
    for k, pwy in pathway_data.items():
        pwy = dogmatize_entities(pwy)
        pwy = dogmatize_edges(pwy)
        result[k] = pwy

    return result


if __name__=="__main__":

    pathway_json = sys.argv[1]
    out_file = sys.argv[2]

    pathway_data = json.load(open(pathway_json, "r"))
    result = dogmatize_all_pathways(pathway_data)

    json.dump(result, open(out_file, "w"))
