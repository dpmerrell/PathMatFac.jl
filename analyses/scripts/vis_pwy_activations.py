
import numpy as np
import h5py
import sys
import os

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import script_util as su


def prepare_df(X, samples, groups, pathways):

    X = X.transpose()
    cols = ["{} ({})".format(s,g) for s, g in zip(samples, groups)]
    samples = np.array(samples)
    groups = np.array(groups)   
    unq_groups = np.unique(groups)

    srt_idx = np.argsort(pathways)
    pathways = pathways[srt_idx]
    X = X[srt_idx,:]

    df = pd.DataFrame(data=X, index=pathways, columns=cols)

    mean_df = df.groupby(by=groups, axis=1).mean()
    mean_df.columns = [col + "_mean" for col in mean_df.columns]
    std_df = df.groupby(by=groups, axis=1).std()
    std_df.columns = [col + "_std" for col in std_df.columns]

    df.loc[:,mean_df.columns] = mean_df.loc[:,:]
    df.loc[:,std_df.columns] = std_df.loc[:,:]

    df["all_mean"] = df.mean(axis=1)
    df["all_std"] = df.std(axis=1)

    # Sort rows by means (for clarity)
    df.sort_values(by=["all_mean"], inplace=True)

    return df 


def ctype_line_plot(df, fig, ctype):

    flag = "("+ctype+")"
    ctype_cols = [col for col in df.columns if col.split(" ")[-1] == flag]

    fig.layout.title = fig.layout.title.text + f"{flag}"

    for col in ctype_cols:
        sample = col.split(" ")[0]
        fig.add_trace(go.Scatter(x=df.index.values, y=df[col].values,
                                 customdata=df.index.values,
                                 hovertemplate="Pathway: %{customdata}<br>Activation: %{y}",
                                 name=sample,
                                 line={"color":"red","width":0.5}
                                )
                     )

    return fig  


def all_ctype_line_plot(df, fig):

    # Build the line plot
    mean_cols = [col for col in df.columns if col.split("_")[-1] == "mean"]
    groups = [col.split("_")[0] for col in mean_cols]
    std_cols = [col + "_std" for col in groups]

    mean_df = df[mean_cols]
    std_df = df[std_cols]
   
    for gp, mc, st in zip(groups, mean_cols, std_cols):
        if gp != "all":
            color="blue"
            width=3
        else:
            color="black"
            width=8
        fig.add_trace(go.Scatter(x=df.index.values, y=df[mc].values,
                                 customdata=df.index.values,
                                 error_y={"type":"data", "array":df[st].values, "visible":True},
                                 hovertemplate="Pathway: %{customdata}<br>Activation: %{y}",
                                 name=gp,
                                 line={"color":color,"width":width}
                                )
                     )

    return fig


def parse_opts(opts, args):

    for arg in args:
        tok = arg.split("=")
        k = tok[0]
        v = tok[1]
        ls = v.split(",")
        opts[k] = ls

    return opts


if __name__=="__main__":

    args = sys.argv
    model_hdf = args[1]
    output_scatter = args[2]

    opts = {"ctype": ["all"]
           }
    if len(args) > 3:
        opts = parse_opts(opts, args[3:])
    ctype = opts["ctype"][0] 
    #exclude_groups = opts["exclude"]
    #keep_groups = opts["keep"]
    #color_col = opts["color"][0]

    # Load information stored in the model
    X = su.load_embedding(model_hdf)
    samples = su.load_sample_ids(model_hdf)
    groups = su.load_sample_groups(model_hdf)
    pathways = su.load_pathway_names(model_hdf)

    df = prepare_df(X, samples, groups, pathways)
    fig = go.Figure(layout={"title":"Pathway Activations"})


    if ctype != "all":
        fig = ctype_line_plot(df, fig, ctype)
    fig = all_ctype_line_plot(df, fig)

    fig.write_html(output_scatter) 
    

