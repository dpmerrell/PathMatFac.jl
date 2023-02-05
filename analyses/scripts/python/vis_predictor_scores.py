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


def plot_prediction_scores(ax, result_data): 
    """

    """

    methods = su.sort_by_order(list(result_data.keys()), su.ALL_METHODS)

    i,j = result_data["idx"]
    nrow, ncol = result_data["N"]
    target = result_data["names"]
    result_jsons = result_data["jsons"] 


    if j == 0:
        ax.set_ylabel(NAMES[rowname])

    return 

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("out_png")
    parser.add_argument("--result_jsons", nargs="+")

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
                       "target":(target_names[j]), 
                       "jsons": dat} for method in method_names[i] } for j, dat in enumerate(row)] for i, row in enumerate(grid)]

    method_names = [NAMES[mn] for mn in method_names]
    target_names = [NAMES[tn] for tn in target_names]
    
    # Plot prediction results across the grid
    fig, axarr = su.make_subplot_grid(plot_prediction_results, grid, 
                                      method_names, target_names)

#    fig.text(0.5, 0.04, "Prediction targets", ha="center")
#    fig.text(0.04, 0.5, "Dimension reduction methods", rotation="vertical", ha="center")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)

 
