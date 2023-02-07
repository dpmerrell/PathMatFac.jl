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

    ymin = 0.0
    ymax = 1.0

    scores = []
    for method in methods:
        dat = result_data[method]
        score_jsons = dat["jsons"]
        all_scores = [json.load(open(rj,"r")) for rj in score_jsons]
        all_scores = [score_dict[score_str] for score_dict in all_scores]
        ymin = min(ymin, np.min(all_scores))
        ymax = max(ymax, np.max(all_scores))
        scores.append(all_scores)
      
    
    method_names = [NAMES[m] for m in methods]
    ax.boxplot(scores, labels=method_names)
    
    y_spread = ymax - ymin
    ax.set_ylim([ymin - 0.05*y_spread, ymax + 0.05*y_spread]) 
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_title(NAMES[target]) 
    ax.set_ylabel(NAMES[score_str])

    return 

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("out_png")
    parser.add_argument("--score_jsons", nargs="+")

    args = parser.parse_args()
    score_jsons = args.score_jsons
    out_png = args.out_png

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

    plt.suptitle("Prediction task performance")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)


 
