
import matplotlib as mpl
import script_util as su
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sys


QTY_SCALES = {"snr": "log",
              "missing": "linear",
              "l1_fraction": "linear",
              "X_pwy_spearman_corr": "linear"
             }


def to_corruption_df(df, x_var, y_var, mode="mean"):

    # Keep only the rows for runs with corrupted pathways 
    df = df.loc[(df["original"] != 1) & (df["randomized"] != 1), :]
    
    # Keep only the columns that define (a) corruption (b) variables of interest and (c) replicates
    corr_columns = ["kadd", "krem", "nadd", "nrem"]
    replicate_columns = ["srep"]
    df = df[corr_columns + [x_var, y_var] + replicate_columns]
    print("DF")
    print(df)

    qmeans = su.aggregate_replicates(df, ["srep", "crep"] + [y_var], op=mode)
    print("QMEANS")
    print(qmeans)
    qmeans["corruption"] = qmeans[["kadd","krem","nadd","nrem"]].apply(lambda r: "_".join(["{:.2f}".format(v) for v in r]), axis=1, result_type="reduce", raw=True)
    
    # Create a new dataframe with index=x_var, columns=corruptions, entries=y_var
    unq_corruptions = list(qmeans["corruption"].unique())
    unq_x = sorted(list(qmeans[x_var].unique()))
    new_df = pd.DataFrame(index=unq_x, columns=unq_corruptions)
    for _, r in qmeans.iterrows():
        new_df.loc[r[x_var], r["corruption"]] = r[y_var]

    return new_df


def make_lineplot(df, out_png, x_var, y_var):

    plt.figure(figsize=(5,3))
    for col in df.columns:
        plt.plot(df.index.values, df[col].values, color="gray", linewidth=0.5)
    
    plt.xlabel(su.NICE_NAMES[x_var])
    plt.ylabel(su.NICE_NAMES[y_var])
    plt.ylim([0,1])
    plt.xscale(QTY_SCALES[x_var])
    plt.yscale(QTY_SCALES[y_var])
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
           
            if i == len(macro_y_vals)-1:
                ax.set_xlabel("{}\n\n$K_r$ = {}".format(su.NICE_NAMES[micro_x_col], psize))
            if j == 0:
                ax.set_ylabel("$K_a$={}\n\n{}".format(myv, su.NICE_NAMES[micro_y_col]))
    
    
    fig.suptitle("{}".format(score_str),fontsize=16)
    plt.tight_layout(rect=[0.0,0.0,1,0.95])
    fig.colorbar(imgs[-1], ax=axarr, location="top", shrink=0.8, pad=0.05, fraction=0.05, use_gridspec=True)

    plt.savefig(output_filename, dpi=300)#, bbox_inches="tight")


if __name__=="__main__":

    args = sys.argv
    scores_tsv = args[1]
    lineplot_png = args[2]
    #grid_png = args[3]
    xvar = args[3]
    yvar = args[4]

    df = pd.read_csv(scores_tsv, sep="\t")
    mean_df = to_corruption_df(df, xvar, yvar, mode="mean")
    print(mean_df) 
    var_df = to_corruption_df(df, xvar, yvar, mode="var")
    make_lineplot(mean_df, lineplot_png, xvar, yvar)

    #grid_df = mean_df.loc[[1],:]
    #grid_df = grid_df.transpose()
    #grid_df.reset_index(inplace=True)
    #grid_df[["kadd","krem","nadd","nrem"]] = grid_df.loc[:,["index"]].apply(lambda x: x["index"].split("_"), axis=1, result_type="expand")
    #print(grid_df)
    #subplot_heatmaps(grid_df, "krem", "kadd", "nrem", "nadd", 1, su.NICE_NAMES[yval],
    #                 output_filename=grid_png)#, vmin=0.0, vmax=1.0)

