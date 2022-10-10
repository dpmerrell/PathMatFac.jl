
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sys

NICE_NAMES = {"nadd": "$N_a$",
              "nrem": "$N_r$",
              "kadd": "$K_a$",
              "krem": "$K_r$"}

def to_snr_corruption_df(df, qoi, mode="mean"):

    # Keep only the scores for runs with
    # corrupted pathways
    df = df.loc[(df["original"] != 1) & (df["randomized"] != 1), :]

    # Group by snr and srep
    gp = df.groupby(["snr","kadd","krem","nadd","nrem"])

    # Determine whether we're computing means or variances
    if mode == "mean":
        qmeans = gp[qoi].mean()
    else:
        qmeans = gp[qoi].var()

    # Reset the index; combine corruption parameters into a single column
    qmeans = pd.DataFrame(qmeans)
    qmeans.reset_index(inplace=True)
    qmeans["corruption"] = qmeans[["kadd","krem","nadd","nrem"]].apply(lambda r: "_".join(["{:.2f}".format(v) for v in r]), axis=1, result_type="reduce", raw=True)
    
    # Create a new dataframe with index=snrs, columns=corruptions
    unq_corruptions = list(qmeans["corruption"].unique())
    unq_snrs = sorted(list(qmeans["snr"].unique()))
    new_df = pd.DataFrame(index=unq_snrs, columns=unq_corruptions)
    for _, r in qmeans.iterrows():
        new_df.loc[r["snr"], r["corruption"]] = r[qoi]

    return new_df

def plot_snr(df, out_png):

    plt.figure(figsize=(5,3))
    for col in df.columns:
        plt.plot(df.index.values, df[col].values, color="gray", linewidth=0.5)
    
    plt.xlabel("SNR")
    plt.ylabel("Pathway Activation Spearman")
    plt.ylim([0,1])
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(out_png)


def make_heatmap(ax, relevant, x_col, y_col, qty_col, **kwargs):

    x_vals = relevant[x_col].unique()
    y_vals = relevant[y_col].unique()

    grid = np.zeros((len(y_vals), len(x_vals)))
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            grid[j,i] = relevant.loc[(relevant[x_col] == x) & (relevant[y_col] == y), qty_col]

    img = ax.imshow(grid, origin="lower", **kwargs)
    ax.set_xticks(list(range(len(x_vals))))
    ax.set_xticklabels(x_vals)
    ax.set_yticks(list(range(len(y_vals))))
    ax.set_yticklabels(y_vals)
    #ax.set_xlim([-0.5, len(x_vals)-0.5])
    #ax.set_ylim([-0.5, len(y_vals)-0.5])

    #ax.label_outer()
    return img


def subplot_heatmaps(qty_df, macro_x_col, macro_y_col, 
                     micro_x_col, micro_y_col, qty_col, score_str,
                     output_filename="simulation_scores.png",
                     macro_x_vals=None, macro_y_vals=None,
                     cmap="Greys", vmin=None, vmax=None):

    if macro_x_vals is None:
        macro_x_vals = sorted(qty_df[macro_x_col].unique().tolist())

    if macro_y_vals is None:
        macro_y_vals = sorted(qty_df[macro_y_col].unique().tolist())[::-1]

    n_rows = len(macro_y_vals)
    n_cols = len(macro_x_vals)

    fig, axarr = plt.subplots(n_rows, n_cols, 
                              sharey=True, sharex=True, 
                              figsize=(2.0*n_cols,2.0*n_rows))
  

    in_macro_y_vals = lambda x: x in macro_y_vals
    relevant_scores = qty_df.loc[qty_df[macro_y_col].map(in_macro_y_vals) , qty_col]
    
    if vmin is None:
        vmin = relevant_scores.quantile(0.05)
    if vmax is None:
        vmax = relevant_scores.quantile(0.95)

    nrm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
    mappable = mpl.cm.ScalarMappable(norm=nrm, cmap=cmap)

    imgs = []

    # Iterate through the different subplots
    for i, myv in enumerate(macro_y_vals):
        for j, psize in enumerate(macro_x_vals):
            
            ax = axarr[i][j]
            relevant = qty_df.loc[(qty_df[macro_y_col] == myv) & (qty_df[macro_x_col] == psize),:]

            img = make_heatmap(ax, relevant, micro_x_col, micro_y_col, qty_col, norm=nrm, cmap=cmap)
            imgs.append(img)
           
            #ax.set_xlim([0,3])
            #ax.set_ylim([0,3])
            if i == len(macro_y_vals)-1:
                ax.set_xlabel("{}\n\n$K_r$ = {}".format(NICE_NAMES[micro_x_col], psize))
            if j == 0:
                ax.set_ylabel("$K_a$={}\n\n{}".format(myv, NICE_NAMES[micro_y_col]))
    
    
    fig.suptitle("{}".format(score_str),fontsize=16)
    plt.tight_layout(rect=[0.0,0.0,1,0.95])
    fig.colorbar(imgs[-1], ax=axarr, location="top", shrink=0.8, pad=0.05, fraction=0.05, use_gridspec=True)

    plt.savefig(output_filename, dpi=300)#, bbox_inches="tight")


if __name__=="__main__":

    args = sys.argv
    scores_tsv = args[1]
    snr_png = args[2]
    grid_png = args[3]
    qoi = args[4]

    df = pd.read_csv(scores_tsv, sep="\t")
    mean_df = to_snr_corruption_df(df, qoi, mode="mean") 
    var_df = to_snr_corruption_df(df, qoi, mode="var")
    plot_snr(mean_df, snr_png)

    grid_df = mean_df.loc[[1],:]
    grid_df = grid_df.transpose()
    grid_df.reset_index(inplace=True)
    grid_df[["kadd","krem","nadd","nrem"]] = grid_df.loc[:,["index"]].apply(lambda x: x["index"].split("_"), axis=1, result_type="expand")
    print(grid_df)
    subplot_heatmaps(grid_df, "krem", "kadd", "nrem", "nadd", 1, "Pathway Spearman Correlation",
                     output_filename=grid_png)#, vmin=0.0, vmax=1.0)

