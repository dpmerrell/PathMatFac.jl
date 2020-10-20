
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
        edge.append(1)
    elif tag[-1] == "|":
        edge.append(-1)

    return edge


def dogmatize_edges(pathway):

    dogmatized = []

    # Add "central dogma" edges
    for ent in pathway["entities"]:
        name_level = ent.split("::")
        name = name_level[0] 
        level = name_level[-1]
        if level == "PROT":
            dogmatized += [[name+"::DNA", name+"::MRNA", 1],
                           [name+"::MRNA", name+"::PROT", 1],
                           [name+"::PROT", name+"::ACT", 1]
                          ]

    # Add the other edges
    for edge in pathway["edges"]:
        dogmatized.append(dogmatize_edge(edge))

    pathway["edges"] = dogmatized

    return pathway


def names_to_idx(edges, all_entities):

    encoder = {v:i for i, v in enumerate(all_entities)}

    idx_edges = []
    for edge in edges:
        idx_edges.append([encoder[edge[0]], encoder[edge[1]], edge[2]])

    return idx_edges


def dogmatize_all_pathways(pathway_data):

    # Get the list of all pathway names
    pwy_names = sorted(list(pathway_data.keys()))

    # "dogmatize" the pathways
    dogmatized = []
    for k in pwy_names:
        pwy = pathway_data[k]
        pwy = dogmatize_entities(pwy)
        pwy = dogmatize_edges(pwy)
        dogmatized.append(pwy)
    
    # Get the list of all pathway entities
    all_entities = set([])
    for pwy in dogmatized:
        all_entities |= set(pwy["entities"])
    all_entities = sorted(list(all_entities))
   
    # Convert the edges from strings to ints
    int_edges = []
    for pwy in dogmatized:
        int_edges.append(names_to_idx(pwy["edges"], all_entities))
 
    result = {"entity_names": all_entities,
              "pathway_names": pwy_names,
              "pathways": int_edges
             }

    return result


if __name__=="__main__":

    pathway_json = sys.argv[1]
    out_file = sys.argv[2]

    pathway_data = json.load(open(pathway_json, "r"))
    result = dogmatize_all_pathways(pathway_data)

    json.dump(result, open(out_file, "w"))
