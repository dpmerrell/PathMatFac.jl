import json
import argparse
import numpy as np

def junk_infer(args):

    with open(args.data_file) as f:
        data = json.load(f)

    with open(args.observation_file) as f:
        observed = json.load(f)
    
    with open(args.pathway_file) as f:
        pathways = json.load(f)

    n_pathways = len(pathways["pathway_names"])
    n_samples = len(data["data"])

    result_means = np.random.randn(n_samples, n_pathways).tolist()
    result_vars = np.random.randn(n_samples, n_pathways)
    result_vars = result_vars**2.0
    result_vars = result_vars.tolist()

    result = {"activations": {
                  "means": result_means,
                  "vars": result_vars
                  }
             }

    with open(args.output_file, "w") as fout:
        json.dump(result, fout)

    return


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="A placeholder that implements a generic interface")
    parser.add_argument("data_file")
    parser.add_argument("observation_file")
    parser.add_argument("pathway_file")
    parser.add_argument("output_file")

    args = parser.parse_args()

    junk_infer(args)
