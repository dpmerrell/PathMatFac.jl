import json
import argparse
import numpy as np

def junk_infer(args):

    with open(args.data_file) as f:
        data = json.load(f)

    with open(args.observation_file) as f:
        observed = json.load(f)
        train_idx = observed["measured_train"]
        test_idx = observed["measured_test"]
    
    with open(args.pathway_file) as f:
        pathways = json.load(f)

    n_pathways = len(pathways["pathway_names"])
    n_samples = len(data["data"])

    activation_means = np.random.randn(n_samples, n_pathways).tolist()
    activation_vars = np.random.randn(n_samples, n_pathways)
    activation_vars = activation_vars**2.0
    activation_vars = activation_vars.tolist()

    result = {"activations": {
                  "means": activation_means,
                  "vars": activation_vars
                  }
             }

    n_test = len(test_idx)
    if n_test > 0:
        test_means = np.random.randn(n_samples, n_test).tolist()
        test_vars = np.random.randn(n_samples, n_test)
        test_vars = test_vars**2.0
        test_vars = test_vars.tolist()
        result["measured_test"] = {"means": test_means,
                                   "vars": test_vars
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
