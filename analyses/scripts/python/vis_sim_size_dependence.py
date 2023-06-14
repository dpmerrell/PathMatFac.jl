
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from copy import copy
import script_util as su
import numpy as np
import pandas as pd
import argparse

mpl.rcParams['font.family'] = "serif"

var_to_N = {0.0125: 1045,
            0.025: 2141,
            0.05: 4108, 
           }

ctypes_to_M = {"CHOL": 36,
               "ESCA": 185,
               "HNSC": 528,
               "HNSC:CESC:ESCA:STAD": 1463,
               "HNSC:CESC:ESCA:STAD:BRCA:LUSC": 3065,
              }
               

def populate_grid(df, field):

    colnames = df.index.levels[0]
    rownames = df.index.levels[1]

    grid = np.zeros((len(rownames), len(colnames)))
    for j,cn in enumerate(colnames):
        for i,rn in enumerate(rownames):
            grid[i,j] = df.loc[(cn,rn)][field]

    rownames = [ctypes_to_M[r] for r in rownames]
    colnames = [var_to_N[c] for c in colnames]

    return rownames, colnames, grid

def annotate_grid(ax, mat, std_mat, color="black"):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            
            ax.text(j, i, "{:.2f}\n({:.2f})".format(mat[i,j], std_mat[i,j]), rotation=0.0, size=5,
                    color=color, horizontalalignment="center", verticalalignment="center")
    return


def plot_means(mean_df, std_df, w=6.0, h=2.5, cm="Greys", vmin=0.3, vmax=1.0):

    fig, axs = plt.subplots(nrows=1, ncols=4, sharey=False, 
                                     gridspec_kw={"width_ratios": [1, 1, 1, 0.1]},
                                     figsize=(w,h))
    # Plot the X span similarities
    rownames, colnames, xsp_mat = populate_grid(mean_df, "X_spansim") 
    _, _, xsp_std = populate_grid(std_df, "X_spansim") 
    print("SPAN SIMILARITY MATRIX")
    print(xsp_mat)
    axs[0].imshow(xsp_mat, aspect="auto", cmap=cm, vmin=vmin, vmax=vmax, origin="lower",
                   interpolation="none")
    axs[0].set_xticks(list(range(len(colnames))))
    axs[0].set_xticklabels(colnames)
    axs[0].set_yticks(list(range(len(rownames))))
    axs[0].set_yticklabels(rownames)
    axs[0].set_ylabel("Samples ($M$)")
    axs[0].set_title("$X$ span sim.")
    annotate_grid(axs[0], xsp_mat, xsp_std, color="white")
    
    # Plot the Y span similarities
    rownames, colnames, ysp_mat = populate_grid(mean_df, "Y_spansim") 
    _, _, ysp_std = populate_grid(std_df, "Y_spansim") 
    axs[1].imshow(ysp_mat, aspect="auto", cmap=cm, vmin=vmin, vmax=vmax, origin="lower",
                   interpolation="none")
    axs[1].set_xticks(list(range(len(colnames))))
    axs[1].set_xticklabels(colnames)
    axs[1].set_xlabel("Features ($N$)")
    axs[1].set_yticks([])
    axs[1].set_yticklabels([])
    axs[1].set_title("$Y$ span sim.")
    annotate_grid(axs[1], ysp_mat, ysp_std)

    # Plot the A_aucpr scores
    rownames, colnames, auc_mat = populate_grid(mean_df, "A_aucpr") 
    _, _, auc_std = populate_grid(std_df, "A_aucpr") 
    axs[2].imshow(auc_mat, aspect="auto", cmap=cm, vmin=vmin, vmax=vmax, origin="lower",
                   interpolation="none")
    axs[2].set_xticks(list(range(len(colnames))))
    axs[2].set_xticklabels(colnames)
    axs[2].set_yticks([])
    axs[2].set_yticklabels([])
    axs[2].set_title("$A$ AUCPR")
    annotate_grid(axs[2], auc_mat, auc_std)

    sm = ScalarMappable(cmap=cm)
    sm.set_clim(vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(mappable=sm, ticks=[vmin, vmax], cax=axs[3], fraction=1.0, orientation="vertical")
    #axs[1][i+1].set_yticks([])
    #axs[1][i+1].set_yticklabels([])
    cbar.set_ticks([vmin, vmax], ["{:.0f}".format(vmin), "{:.0f}".format(vmax)])
    fig.suptitle("Factor recovery vs. problem size")

    return fig

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("result_tsv")
    parser.add_argument("out_png")

    args = parser.parse_args()

    result_tsv = args.result_tsv
    out_png = args.out_png

    df = pd.read_csv(result_tsv, sep="\t")
    size_df = df.loc[:,["kept_ctypes", "var_filter","rep","Y_spansim","X_spansim","A_aucpr"]]
    size_df = size_df.loc[df["max_epochs"] == 1000]

    size_gb = size_df.groupby(["var_filter", "kept_ctypes"])
    size_mean_df = size_gb.mean()
    size_std_df = size_gb.std()
    print("MEAN DF:")
    print(size_mean_df)

    print(size_std_df)
    f = plot_means(size_mean_df, size_std_df)
    f.tight_layout(h_pad=0.01, w_pad=0.01,pad=0.5)
    f.savefig(out_png, dpi=300)

