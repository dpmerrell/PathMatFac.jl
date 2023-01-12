
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import script_util as su
import pandas as pd
import numpy as np
import h5py
import sys


def get_mrnaseq_df(model_hdf):

    features = su.load_features(model_hdf)
    Y = su.load_feature_factors(model_hdf)
    pwy_names = su.load_pathway_names(model_hdf)

    feature_assays = np.vectorize(lambda x: x.split("_")[-1])(features) 
    feature_genes = np.vectorize(lambda x: x.split("_")[0])(features) 
    mrnaseq_idx = (feature_assays == "mrnaseq")

    # Some transformations on Y 
    Y = Y.transpose()
    medians = np.median(Y, axis=1)
    is_negative = (medians < 0.0)
    Y[is_negative,:] = -1.0*Y[is_negative,:]
    Y = Y[:,mrnaseq_idx]
    feature_genes = feature_genes[mrnaseq_idx]

    mrnaseq_df = pd.DataFrame(data=Y, index=pwy_names, columns=feature_genes)
    return mrnaseq_df


def plot_heatmap(mrnaseq_df):

    fig = px.imshow(mrnaseq_df, aspect="auto", 
                    color_continuous_scale="RdBu_r",
                    color_continuous_midpoint=0.0)
    return fig


if __name__=="__main__":

    model_hdf = sys.argv[1]
    out_html = sys.argv[2]

    mrnaseq_df = get_mrnaseq_df(model_hdf)

    fig = plot_heatmap(mrnaseq_df)

    fig.write_html(out_html)     
