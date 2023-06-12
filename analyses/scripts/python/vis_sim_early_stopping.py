
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from copy import copy
import script_util as su
import numpy as np
import pandas as pd
import argparse

mpl.rcParams['font.family'] = "serif"


def plot_lines(mean_df, std_df, y_vars, w=5, h=3): 

    fig = plt.figure(figsize=(w,h))
    x = list(range(mean_df.shape[0]))
    x_ticklabels = mean_df.index.values
    for i, yv in enumerate(y_vars):
        y = mean_df[yv].values
        y_std = std_df[yv].values
        plt.errorbar(x, y, yerr=y_std, linewidth=1.0, elinewidth=0.5, mew=0.5, capsize=3.0, label=su.NICE_NAMES[yv], color=su.ALL_COLORS[i])

    plt.legend(loc="lower right")

    plt.ylim([0,1])
    plt.xlim([-0.5,len(x)-0.5])

    plt.xticks(x, x_ticklabels)
    plt.title("Robustness to early stopping")

    plt.xlabel("Training iterations")
    plt.ylabel("Factor recovery scores")

    return fig


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("result_tsv")
    parser.add_argument("out_png")

    args = parser.parse_args()

    result_tsv = args.result_tsv
    out_png = args.out_png

    df = pd.read_csv(result_tsv, sep="\t")
    df = df.loc[df["kept_ctypes"] == "HNSC:CESC:ESCA:STAD"]
    df = df.loc[df["var_filter"] == 0.05]
    df = df.loc[:,["max_epochs","Y_spansim","X_spansim","A_aucpr"]]

    gb = df.groupby(["max_epochs"])
    mean_df = gb.mean()
    std_df = gb.std()
    print("MEAN DF:")
    print(mean_df)
    print(std_df)

    f = plot_lines(mean_df, std_df, ["X_spansim", "Y_spansim", "A_aucpr"])
    f.tight_layout(h_pad=0.01, w_pad=0.01,pad=0.5)
    f.savefig(out_png, dpi=300)

