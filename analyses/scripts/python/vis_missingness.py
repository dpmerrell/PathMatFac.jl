

from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from copy import copy
import script_util as su
import numpy as np
import json
import argparse

mpl.rcParams['font.family'] = "serif"


def plot_losses(stage1_losses, stage2_losses, w=6, h=3): 

    fig, axarr = plt.subplots(2, 1, figsize=(w,h), sharex=True)

    #x = list(range(len(stage1_losses[0])))
    #x_ticklabels = mean_df.index.values
    x = np.arange(1, 996)
    x_ticks = np.array([5,10,25,50,125,250,500,1000]) - 4
    x_ticklabels = np.array([5,10,25,50,125,250,500,1000])

    for l in stage1_losses:
        axarr[0].plot(x, l, "k", linewidth=0.75)

    axarr[0].set_yscale("log") 
    axarr[0].set_ylabel("Stage 1 Loss") 

    for l in stage2_losses:
        axarr[1].plot(x, l, "k", linewidth=0.75)

    axarr[1].set_xlim([5,995])
    axarr[1].set_xlabel("Training iterations")
    axarr[1].set_xscale("log")
    axarr[1].set_xticks(x_ticks)
    axarr[1].set_xticklabels(x_ticklabels)
 
    axarr[1].set_yscale("log") 
    axarr[1].set_ylabel("Stage 2 Loss") 

    fig.suptitle("Training loss (log-log)")     
    #for i, yv in enumerate(y_vars):
    #    y = mean_df[yv].values
    #    y_std = std_df[yv].values
    #    plt.errorbar(x, y, yerr=y_std, linewidth=1.0, elinewidth=0.5, mew=0.5, capsize=3.0, label=su.NICE_NAMES[yv], color=su.ALL_COLORS[i])

    #plt.legend(loc="lower right")

    #plt.ylim([0,1])
    #plt.xlim([-0.125,len(x)-0.875])

    #plt.xticks(x, x_ticklabels)
    #plt.title("Robustness to missing data")

    #plt.xlabel("Fraction of values missing")
    #plt.ylabel("Factor recovery scores")

    return fig


def extract_mf_fits(history_ls):

    fit_ls = []
    cur_fit = []
    in_fit = False
    for item in history_ls:
        # Detect the start of a MF fit
        if item["name"] == "mf_fit_lr=1.0":
            in_fit = True
            cur_fit = []
            fit_ls.append(cur_fit)
        # Detect the end of a MF fit
        elif "mf_fit" not in item["name"]:
            in_fit = False

        # If we are on a "mf_fit" item, then concatenate
        # the losses from this item
        if in_fit:
            cur_fit += list(item["total_loss"])

    return fit_ls


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("out_png")
    parser.add_argument("--history_jsons", nargs="+")

    args = parser.parse_args()

    history_jsons = args.history_jsons
    out_png = args.out_png

    histories = [json.load(open(js, "r")) for js in history_jsons]

    fits = [extract_mf_fits(h) for h in histories]
    stage1_losses = [f[-2][5:] for f in fits]
    stage2_losses = [f[-1][5:] for f in fits]

    f = plot_losses(stage1_losses, stage2_losses)

    f.tight_layout(h_pad=0.01, w_pad=0.01,pad=0.5)
    f.savefig(out_png, dpi=300)

