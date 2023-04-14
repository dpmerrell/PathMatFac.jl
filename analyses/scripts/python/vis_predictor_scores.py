# vis_prediction_scores.py
#
# Visualize prediction performance 
# across methods and tasks.

from matplotlib import pyplot as plt
import matplotlib as mpl
import script_util as su
from os import path
import pandas as pd
import numpy as np
import argparse
import json

NAMES = su.NICE_NAMES
SCORE_MAP = su.ALL_SCORES


def plot_prediction_scores(ax, result_data): 
    """

    """
    methods = su.sort_by_order(list(result_data.keys()), su.ALL_METHODS)
    target = result_data[methods[0]]["target"]
    score_str = SCORE_MAP[target] 
    trivial_score_str = score_str + "_baseline"

    ymin = 0.0
    ymax = 1.0
    xmin = 0.5
    xmax = len(methods) + 0.5

    all_scores = []
    all_trivial_scores = set()
    for method in methods:
        dat = result_data[method]
        score_jsons = dat["jsons"]
        score_dicts = [json.load(open(rj,"r")) for rj in score_jsons]
        trivial_scores = [score_dict[trivial_score_str] for score_dict in score_dicts]
        scores = [score_dict[score_str] for score_dict in score_dicts]
        rel_scores = [s - t for s,t in zip(scores, trivial_scores]
        ymin = min(ymin, np.min(rel_scores))
        ymax = max(ymax, np.max(rel_scores))
        all_scores.append(rel_scores)
        #all_trivial_scores |= set(trivial_scores)

    #for ts in list(all_trivial_scores):
    #    ax.plot([xmin, xmax],[ts, ts], "--", linewidth=0.5, color="grey")
 
    method_names = [NAMES[m] for m in methods]
    ax.boxplot(all_scores, labels=method_names)
    
    y_spread = ymax - ymin
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin - 0.05*y_spread, ymax + 0.05*y_spread]) 
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_title(NAMES[target]) 
    ax.set_ylabel(NAMES[score_str])

    return 

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("out_png")
    parser.add_argument("--score_jsons", nargs="+")
    parser.add_argument("--n_folds", type=int, default=5)

    args = parser.parse_args()
    score_jsons = args.score_jsons
    out_png = args.out_png
    n_folds = args.n_folds

    # Arrange the JSON filepaths into a grid,
    # indexed by (method, target)
    method_target_jsons = su.get_methods_targets(score_jsons)
    grid, method_names, target_names = su.dict_to_grid(method_target_jsons)
    nrow = len(method_names)
    ncol = len(target_names)
    
    # Add some auxiliary data to the grid
    grid = [[{method: {"idx":(i,j), 
                       "N": (nrow, ncol), 
                       "target": target, 
                       "jsons": grid[i][j]} for i, method in enumerate(method_names)} for j, target in enumerate(target_names)]] 

    target_names = [NAMES[tn] for tn in target_names]
    
    figsize=(3.0*ncol, 4.0)

    # Plot prediction results across the grid
    fig, axarr = su.make_subplot_grid(plot_prediction_scores, grid, 
                                      [1], target_names, figsize=figsize)

    plt.suptitle("Performance relative to trivial baseline ({}-fold cross-validation)".format(n_folds))
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)


 
