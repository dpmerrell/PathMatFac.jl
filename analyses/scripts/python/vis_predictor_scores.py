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
    score_str = SCORE_MAP[result_data[methods[0]["target"]]] 

    scores = []
    for method in methods:
        dat = result_data[method]
        result_jsons = dat["jsons"]
        all_scores = [json.load(open(rj,"r")) for rj in result_jsons]
        scores.append([score_dict[score_str] for score_dict in all_scores])
        
    ax.boxplot(scores, labels=methods)
    ax.set_ylabel(NAMES[score_str])

    return 

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("out_png")
    parser.add_argument("--score_jsons", nargs="+")

    args = parser.parse_args()
    result_jsons = args.result_jsons
    out_png = args.out_png

    # Arrange the JSON filepaths into a grid,
    # indexed by (method, target)
    method_target_jsons = su.get_methods_targets(result_jsons)
    grid, method_names, target_names = su.dict_to_grid(method_target_jsons)
    nrow = len(method_names)
    ncol = len(target_names)
    
    # Add some auxiliary data to the grid
    grid = [[{method: {"idx":(i,j), 
                       "N": (nrow, ncol), 
                       "target": target, 
                       "jsons": grid[i][j]} for i, method in enumerate(method_names))} for j, target in enumerate(target_names)]] 

    target_names = [NAMES[tn] for tn in target_names]
    
    # Plot prediction results across the grid
    fig, axarr = su.make_subplot_grid(plot_prediction_scores, grid, 
                                      [1], target_names)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)


 
