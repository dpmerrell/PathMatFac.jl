
import argparse
import sys
import numpy as np
import json


def random_junk_sim(args):

    with open(args.pathway_file) as f:
        pathways = json.load(f)
    
    n_entities = len(pathways["entity_names"])
    n_pathways = len(pathways["pathway_names"])

    activations = np.random.randn(args.n_samples, n_pathways)
    data = np.random.randn(args.n_samples, n_entities)

    result = {"activations": activations.tolist(),
              "data": data.tolist()
              }
    
    with open(args.output_file, "w") as f:
        json.dump(result, f)

    return



def simulate_data(args):

    if args.method == "random_junk":
        random_junk_sim(args)

    return



if __name__=="__main__":


    parser = argparse.ArgumentParser(description="wrapper around data simulation scripts")
    parser.add_argument("method", type=str,
                         choices=["random_junk"], 
                         help="the method used to simulate data")
    parser.add_argument("pathway_file", type=str,
                         help="path to a JSON file containing preprocessed pathway information")
    parser.add_argument("n_samples", type=int,
                         help="number of samples (rows) in the generated data")
    parser.add_argument("output_file", type=str,
                         help="path for the output JSON file") 
    args = parser.parse_args()
    print(args.output_file)
    simulate_data(args)


