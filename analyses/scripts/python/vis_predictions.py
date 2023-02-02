
from matplotlib import pyplot as plt
import matplotlib as mpl
import script_util as su
from os import path
import pandas as pd
import numpy as np
import argparse


def plot_roc_curves(ax, result_jsons):

    if len(result_jsons) == 0:
        return

    result_data = [json.load(open(js), "r") for js in result_jsons]
    for d in result_data:
        fpr = d["fpr"]
        tpr = d["tpr"]

        ax.plot(fpr, tpr, linewidth=0.25, linecolor="grey")

    return


def plot_confusion_matrix(ax, result_jsons):

    if len(result_jsons) == 0:
        return

    result_data = [json.load(open(js), "r") for js in result_jsons]
    
    N = len(result_data[0]["classes"])
    full_confusion = np.zeros((N,N))
    for d in result_data:
        conf = np.array(d["confusion"])
        full_confusion += conf

    ax.imshow(full_confusion)
    return


def plot_regression_scatter(ax, result_jsons):

    if len(result_jsons) == 0:
        return

    result_data = [json.load(open(js), "r") for js in result_jsons]
    y_true = [d["y_true"] for d in result_data]
    y_true = np.concatenate(*y_true)

    y_pred = [d["y_pred"] for d in result_data]
    y_pred = np.concatenate(*y_pred)

    ax.scatter(y_true, y_pred, color="grey", s=0.5)

    return


def plot_ordinal_scatter(ax, result_jsons):

    if len(result_jsons) == 0:
        return

    result_data = [json.load(open(js), "r") for js in result_jsons]
    y_true = [d["y_true"] for d in result_data]
    y_true = np.concatenate(*y_true)

    # Add some jitter to the horizontal axis
    N = length(y_true)
    y_true += 0.01*np.random.randn(N)
    
    y_pred = [d["y_pred"] for d in result_data]
    y_pred = np.concatenate(*y_pred)

    ax.scatter(y_true, y_pred, color="grey", s=0.5)

    return


def plot_prediction_results(ax, result_data): 
    """

    """

    i,j = result_data["idx"]
    nrow, ncol = result_data["N"]
    rowname, colname = result_data["names"]
    result_jsons = result_data["jsons"] 
   
    if colname in ("hpv_status"):
        plot_roc_curves(ax, result_jsons)
    elif colname in ("ctype"):
        plot_confusion_matrix(ax, result_jsons)
    elif colname in ("survival"):
        plot_survival_scatter(ax, result_jsons)
    elif colname in ("pathologic_stage"):
        plot_ordinal_scatter(ax, result_jsons)

    if i == n_row - 1:
        ax.set_xlabel(colname)
    if j == 0:
        ax.set_ylabel(rowname)

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
    grid = [[{"idx":(i,j), 
              "N": (nrow, ncol), 
              "names":(method_names[i],target_names[j]), 
              "jsons": dat} for j, dat in enumerate(row)] for i, row in enumerate(col)]

    # Plot prediction results across the grid
    method_names = [su.NICE_NAMES[mn] for mn in method_names]
    target_names = [su.NICE_NAMES[tn] for tn in target_names]
    
    fig, axarr = su.make_subplot_grid(plot_prediction_results, grid, 
                                      method_names, target_names)

    plt.savefig(out_png, dpi=300)

 
