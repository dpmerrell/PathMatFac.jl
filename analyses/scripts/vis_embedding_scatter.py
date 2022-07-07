
import numpy as np
import h5py
import sys
import os

import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

import script_util as su


def load_clinical_data(clinical_hdf, cols):
    
    with h5py.File(clinical_hdf, "r") as f:
        clinical_data = f["/data"][:,:].transpose()

    sample_ids = su.load_hdf(clinical_hdf, "/columns", dtype=str)
    features = su.load_hdf(clinical_hdf, "/index", dtype=str)

    col_to_idx = su.value_to_idx(features)
    idx = [col_to_idx[col] for col in cols]

    return clinical_data[:,idx].astype(str), sample_ids



def match_clinical_to_omic(clinical_data, clinical_samples, orig_samples):

    M = orig_samples.shape[0]
    L = clinical_data.shape[1]
    new_clinical_data = np.zeros((M,L), dtype=np.object0)
    new_clinical_data[:,:] = "NA"

    clinical_idx, sample_idx = su.keymatch(clinical_samples, orig_samples)

    new_clinical_data[sample_idx,:] = clinical_data[clinical_idx,:]
    new_clinical_data[new_clinical_data=="nan"] = "NA"

    return new_clinical_data


def compute_pca(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X = (X - mu)/sigma

    u, s, vh = np.linalg.svd(X, full_matrices=False)

    return u, s, vh


def get_pc_names(n_pc):
    return np.array(["PC {}".format(idx+1) for idx in range(U.shape[1])])


def embedding_scatter(U, instance_ids, instance_groups, 
                      clinical_data, clinical_cols, pc_idx,
                      color_col="CancerType"):

    labels = get_pc_names(U.shape[1])

    source_df = pd.DataFrame(data=U, columns=labels)
    source_df["CancerType"] = instance_groups
    source_df["Patient ID"] = instance_ids

    hover_cols = []

    for i, col in enumerate(clinical_cols):
        col_name = su.NICE_NAMES[col]
        source_df[col_name] = clinical_data[:,i]
        hover_cols.append(col_name)

    fig = px.scatter_3d(source_df, x=labels[pc_idx[0]], 
                                   y=labels[pc_idx[1]], 
                                   z=labels[pc_idx[2]],
                                   color=color_col,
                                   hover_data=["CancerType"]+hover_cols,
                                   title="Pathway Embedding",
                       )
    fig.update_traces(marker=dict(line=dict(width=2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers')) 

    return fig


def pc_line_plot(Vh, pathways, pc_idx):
    
    pc_names = get_pc_names(Vh.shape[0])
    Vh = Vh.transpose()

    used_pcs = pc_names[pc_idx]
    df = pd.DataFrame(data=Vh[:,pc_idx], columns=used_pcs)
    df["Pathway"] = pathways
    df.sort_values(["Pathway"], inplace=True)
    df.index = list(range(Vh.shape[0]))
    fig = px.line(data_frame=df, y=used_pcs, hover_data=["Pathway"],
                      labels={"index": "Pathway ID",
                              "variable": "Principal Component",
                              "value": "Weight"
                             },
                      title="Principal Components"
                 )
    return fig


def explained_var_plot(s):
    pc_names = get_pc_names(len(s))

    df = pd.DataFrame(data=s.reshape((len(s),1)),columns=["Singular Value"])
    df["Principal Component"] = pc_names

    fig = px.line(data_frame=df, y=["Singular Value"],
                  labels={"index": "Principal Component",
                          "variable": "Singular Value",
                          }
                 )

    return fig



def combine_figs(scatter_fig, explained_var_fig, line_fig):
    fig = make_subplots(
        rows=2,
        cols=2,
        row_heights=[0.8, 0.2],
        column_widths=[0.2, 0.8],
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
        subplot_titles=["Embedding Scatterplot",
                        "Explained Variance",
                        "Top Principal Components"],
        specs=[[{"type": "scatter3d", "colspan":2}, None], 
               [{"type": "scatter"}, {"type": "scatter"}]]
    )

    fig.add_traces(scatter_fig.data, rows=1, cols=1) 
    fig.add_traces(explained_var_fig.data, rows=2, cols=1)
    fig.add_traces(line_fig.data, rows=2, cols=2) 

    fig.update_layout({'scene': {
                                'xaxis': {'title': {'text': 'PC 1'}},
                                'yaxis': {'title': {'text': 'PC 2'}},
                                'zaxis': {'title': {'text': 'PC 3'}}
                               }
                      })


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
    clinical_hdf = args[2]
    first_pc = int(args[3])
    output_scatter = args[4]

    opts = {"exclude": [],
            "keep": [],
            "color": ["CancerType"]
           }
    if len(args) > 5:
        opts = parse_opts(opts, args[5:])

    exclude_groups = opts["exclude"]
    keep_groups = opts["keep"]
    color_col = opts["color"][0]

    # Load information stored in the model
    orig_X = su.load_embedding(model_hdf)
    orig_samples = su.load_sample_ids(model_hdf)
    orig_groups = su.load_sample_groups(model_hdf)
    pathways = su.load_pathway_names(model_hdf)

    # Filter out the excluded groups
    if len(keep_groups) > 0:
        kept_idx = np.vectorize(lambda x: x in keep_groups)(orig_groups)
    else:
        kept_idx = np.vectorize(lambda x: x not in exclude_groups)(orig_groups)

    X = orig_X[kept_idx,:]
    samples = orig_samples[kept_idx]
    groups = orig_groups[kept_idx]

    # Compute principal components
    U, s, Vh = compute_pca(X)

    clinical_cols = ["gender", "hpv_status"] 
    clinical_data, clinical_samples = load_clinical_data(clinical_hdf, clinical_cols)
    clinical_data = match_clinical_to_omic(clinical_data, clinical_samples, samples)
    pc_idx = list(range(first_pc, first_pc+3))

    scatter_fig = embedding_scatter(U, samples, groups, clinical_data, clinical_cols, pc_idx,
                                    color_col=color_col)
    explained_var_fig = explained_var_plot(s)
    line_fig = pc_line_plot(Vh, pathways, pc_idx)

    fig = combine_figs(scatter_fig, explained_var_fig, line_fig)
    fig.write_html(output_scatter) 


