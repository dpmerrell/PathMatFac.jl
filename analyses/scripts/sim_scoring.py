import json
import argparse
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import spearmanr


def score_activations(args):

    with open(args.inferred) as f:
        inferred = json.load(f)

    with open(args.simulated) as f:
        simulated = json.load(f)

    true_acts = simulated["activations"]
    inferred_means = inferred["activations"]["means"]
    n_patients = len(true_acts)    

    scores = [spearmanr(true_acts[i], inferred_means[i])[0] for i in range(n_patients) ]

    result = {"spearman": scores}

    return result


def score_test_pred(args):
    
    with open(args.obs_pattern) as f:
        obs_pattern = json.load(f)
        test_idx = obs_pattern["measured_test"]

    if len(test_idx) == 0:
        return {}

    with open(args.inferred) as f:
        inferred = json.load(f)

    with open(args.simulated) as f:
        simulated = json.load(f)

    true_measurements = np.array(simulated["data"])
    true_test = true_measurements[:,test_idx]

    inferred_means = np.array(inferred["activations"]["means"])

    score = r2_score(true_test, inferred_means, multioutput="variance_weighted")
    result = {"r2": score}

    return result


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Scores inferred pathway activations against true (simulated) ones")
    parser.add_argument("inferred")
    parser.add_argument("simulated")
    parser.add_argument("obs_pattern")
    parser.add_argument("out_file")

    args = parser.parse_args()
    act_score = score_activations(args)
    test_pred_score = score_test_pred(args)

    result = {"activations": act_score,
              "measured_test": test_pred_score
             }

    with open(args.out_file, "w") as f:
        json.dump(result, f)

