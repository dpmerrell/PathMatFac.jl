

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import matplotlib as mpl
import script_util as su
from os import path
import pandas as pd
import numpy as np
import argparse
import h5py

NAMES = su.NICE_NAMES

mpl.rcParams['font.family'] = "serif"

def plot_batch_scores(df, score="logdelta_spearman"): 
    """

    """
    ymin = 0.0
    ymax = 1.0

    unq_wbs = np.sort(np.unique(df["within_batch_std"].values))
    unq_bbs = np.sort(np.unique(df["between_batch_std"].values))
    #unq_methods = np.sort(np.unique(df["batch_method"].values))
    unq_methods = ["EM", "LSQ"]
    nice_methods = [NAMES[n] for n in unq_methods]
    xmin = 0.5
    xmax = len(unq_methods) + 0.5

    fig, axarr = plt.subplots(len(unq_wbs), len(unq_bbs), figsize=(5,5), sharey=True, sharex=True)

    for i, wbs in enumerate(unq_wbs):
        for j, bbs in enumerate(unq_bbs):
            all_scores = []
            for method in unq_methods:
                rel_data = df.loc[(df["within_batch_std"] == wbs) & (df["between_batch_std"] == bbs) & (df["batch_method"] == method), score].values
                all_scores.append(rel_data)
            
            ax = axarr[i][j]
            ax.boxplot(all_scores, labels=unq_methods)
            #ax.set_ylim([0,1])

            if j == 0:
                ax.set_ylabel("w.b.std. {}".format(wbs))
            #if j > 0:
            #    ax.set_yticks([])

            if i == len(unq_bbs) - 1:
                ax.set_xlabel("b.b.std. {}".format(bbs))
            #if i < len(unq_bbs) - 1:
            #    ax.set_xticks([])

    fig.suptitle("Batch scale Spearman correlation")
    
    #axarr[1][0].text(-1.0, 0.825, "Within-batch std.", rotation=90.0, size=12.0, 
    #                 horizontalalignment="right", verticalalignment="center")
    #axarr[2][1].text(1.5, -1.0, "Between-batch std.", rotation=0.0, size=12.0, 
    #                 horizontalalignment="center", verticalalignment="top")
    #ax[2][1].annotate("Between-batch std.") 

    print("WITHIN BATCH STDS")
    print(unq_wbs)
    print("BETWEEN BATCH STDS")
    print(unq_wbs)
    #ax.boxplot(all_scores, labels=method_names)
    
    #ax.set_xlim([xmin, xmax])
    #ax.tick_params(axis='x', labelrotation=45)
    #ax.set_title(NAMES[target])

    return fig 


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("score_tsv")
    parser.add_argument("out_png")

    args = parser.parse_args()
    score_tsv = args.score_tsv
    out_png = args.out_png

    df = pd.read_csv(score_tsv, sep="\t")
    df = df.loc[:,["within_batch_std","between_batch_std", "batch_method", "theta_r2","logdelta_spearman"]]

    fig = plot_batch_scores(df)
    #gp = df.groupby(["within_batch_std","between_batch_std", "batch_method"])
    #mean_df = gp.mean()
    #std_df = gp.std()

    #print("MEAN DF")
    #print(mean_df)

    #print("STD DF")
    #print(std_df)

    # Plot prediction results across the grid
    #fig, axarr = su.make_subplot_grid(plot_prediction_embeddings, grid, 
    #                                  method_names, target_names)

    #fig.text(0.5, 0.04, "Prediction targets", ha="center")
    #fig.text(0.04, 0.5, "Dimension reduction methods", rotation="vertical", ha="center")
    plt.tight_layout(h_pad=0.05, w_pad=0.1)
    plt.savefig(out_png, dpi=300)

 
