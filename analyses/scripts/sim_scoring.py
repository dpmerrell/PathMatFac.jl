import json
import argparse
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

    result = {"activation": {"spearman": scores} }

    with open(args.out_file, "w") as f:
        json.dump(result, f)
    return



if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Scores inferred pathway activations against true (simulated) ones")
    parser.add_argument("inferred")
    parser.add_argument("simulated")
    parser.add_argument("out_file")

    args = parser.parse_args()
    score_activations(args)


