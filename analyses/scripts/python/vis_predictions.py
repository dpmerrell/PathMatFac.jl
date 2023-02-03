
from matplotlib import pyplot as plt
import matplotlib as mpl
import script_util as su
from os import path
import pandas as pd
import numpy as np
import argparse
import json

NAMES = su.NICE_NAMES

def plot_roc_curves(ax, result_jsons):

    if len(result_jsons) == 0:
        return

    result_data = [json.load(open(js, "r")) for js in result_jsons]

    aucs = []
    for d in result_data:
        fpr = d["roc"]["fpr"]
        tpr = d["roc"]["tpr"]

        ax.plot(fpr, tpr, linewidth=0.5, color="grey")
        aucs.append(np.trapz(tpr, x=fpr)

    ax.plot([0,1],[0,1], "k--", linewidth=1.0)
    ax.set_xlim([-0.05,1])
    ax.set_ylim([0,1.05])
    ax.annotate("Mean AUC: {}".format(np.mean(aucs)), (0.75,0.25))

    return


def plot_confusion_matrix(ax, result_jsons):

    if len(result_jsons) == 0:
        return

    result_data = [json.load(open(js, "r")) for js in result_jsons]
    
    N = len(result_data[0]["classes"])
    full_confusion = np.zeros((N,N))
    for d in result_data:
        conf = np.array(d["confusion"])
        full_confusion += conf

    ax.imshow(full_confusion, origin="upper", cmap="Greys")
    return


def plot_regression_scatter(ax, result_jsons):

    if len(result_jsons) == 0:
        return

    result_data = [json.load(open(js, "r")) for js in result_jsons]
    y_true = [d["y_true"] for d in result_data]
    y_true = np.concatenate(y_true)

    y_pred = [d["y_pred"] for d in result_data]
    y_pred = np.concatenate(y_pred)

    ax.scatter(y_true, y_pred, color="grey", s=0.5)

    return


def plot_ordinal_scatter(ax, result_jsons):

    if len(result_jsons) == 0:
        return

    result_data = [json.load(open(js, "r")) for js in result_jsons]
    y_true = [d["y_true"] for d in result_data]
    y_true = np.concatenate(y_true)

    # Add some jitter to the horizontal axis
    N = len(y_true)
    y_true += 0.05*np.random.randn(N)
    
    y_pred = [d["y_pred"] for d in result_data]
    y_pred = np.concatenate(y_pred)

    ax.scatter(y_true, y_pred, color="grey", s=0.5)

    return


def plot_survival_scatter(ax, result_jsons):
    
    result_data = [json.load(open(js, "r")) for js in result_jsons]
    true_survival = np.concatenate([np.array(d["true_survival"]) for d in result_data]).astype(float)
    pred_survival = np.concatenate([np.array(d["pred_survival"]) for d in result_data]).astype(float)

    dtd = true_survival[:,0]
    dtlf = true_survival[:,1]
    is_alive = np.isnan(dtd)
    is_dead = np.logical_not(is_alive)

    true_dead = dtd[is_dead]
    pred_dead = pred_survival[is_dead]

    true_alive = dtlf[is_alive]
    pred_alive = pred_survival[is_alive]
 
    ax.scatter(true_alive, pred_alive, s=0.25, marker='o', color='blue')
    ax.scatter(true_dead, pred_dead, s=0.75, marker='x', color='red')

    return


def plot_prediction_results(ax, result_data): 
    """

    """

    i,j = result_data["idx"]
    nrow, ncol = result_data["N"]
    rowname, colname = result_data["names"]
    result_jsons = result_data["jsons"] 

    print(rowname, colname)   

    if colname in ("hpv_status"):
        plot_roc_curves(ax, result_jsons)
    elif colname in ("ctype"):
        plot_confusion_matrix(ax, result_jsons)
    elif colname in ("survival"):
        plot_survival_scatter(ax, result_jsons)
    elif colname in ("pathologic_stage"):
        plot_ordinal_scatter(ax, result_jsons)

    if i == nrow - 1:
        ax.set_xlabel(NAMES[colname])
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
    grid = [[{"idx":(i,j), 
              "N": (nrow, ncol), 
              "names":(method_names[i],target_names[j]), 
              "jsons": dat} for j, dat in enumerate(row)] for i, row in enumerate(grid)]

    method_names = [NAMES[mn] for mn in method_names]
    target_names = [NAMES[tn] for tn in target_names]
    
    # Plot prediction results across the grid
    fig, axarr = su.make_subplot_grid(plot_prediction_results, grid, 
                                      method_names, target_names)

    fig.text(0.5, 0.04, "Prediction targets", ha="center")
    fig.text(0.04, 0.5, "Dimension reduction methods", rotation="vertical", ha="center")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)

 
