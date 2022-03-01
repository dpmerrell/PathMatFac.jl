

from plotly import graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import script_util as su
import pandas as pd
import numpy as np
import h5py
import sys


def make_batch_heatmap(batch_df, value_df, cancer_types, color_scale="RdBu_r", color_midpoint=0.0,
                       zmin=None, zmax=None):

    fig = px.imshow(value_df.values, x=value_df.columns,
                    aspect="auto", color_continuous_scale=color_scale,
                    color_continuous_midpoint=color_midpoint,
                    zmin=zmin, zmax=zmax)

    ctype_array = np.repeat(cancer_types[:,np.newaxis], 
                            value_df.shape[1], axis=1)

    fig.data[0].customdata = np.dstack((ctype_array, batch_df.values))
    fig.data[0].hovertemplate = "cancer: %{customdata[0]}<br>batch: %{customdata[1]}<br>assay: %{x}<br>value: %{z}<extra></extra>"
    fig.data[0].hoverongaps = False

    return fig


def col_param_plot(param, col_names):
    
    fig = px.line(param)
    fig.data[0].customdata = col_names
    fig.data[0].hovertemplate = "column: %{customdata}<br>value: %{y}<extra></extra>"

    return fig



assay_ordering = ["methylation", "mrnaseq", "rppa", "cna", "mutation"]
assay_to_order = {a:idx for idx, a in enumerate(assay_ordering)}


def make_batch_dfs(feature_batch_ids, sample_batch_ids, value_dicts, internal_sample_idx):

    fb_to_idx = {fb:idx for idx, fb in enumerate(feature_batch_ids)} 
    valid_fb_idx = [fb_to_idx[assay] for assay in assay_ordering]
    valid_fb = [feature_batch_ids[i] for i in valid_fb_idx]
    valid_sb = [sample_batch_ids[i][internal_sample_idx] for i in valid_fb_idx]
    valid_vd = [value_dicts[i] for i in valid_fb_idx]

    batches_arr = np.array(valid_sb).transpose()   

    batch_df = pd.DataFrame(data=batches_arr, columns=valid_fb) 
    empty_idx = (batch_df == "")

    values_arr = np.zeros(batches_arr.shape)
    values_arr[...] = np.nan
    for j, vd in enumerate(valid_vd):
        values_arr[:,j] = np.vectorize(lambda x: vd[x])(batches_arr[:,j])
    values_arr[empty_idx] = np.nan

    values_df = pd.DataFrame(data=values_arr, columns=valid_fb)

    return batch_df, values_df


def combine_figs(theta_fig, delta_fig, mu_fig, sigma_fig):

    fig = make_subplots(rows=2,cols=2,
                        row_heights=[0.2, 0.8],
                        vertical_spacing=0.1,
                        horizontal_spacing=0.05,
                        subplot_titles=["Column Shift","Column Scale",
                                        "Batch Shift","Batch Scale"],
                        )
 
    fig.add_traces(mu_fig.data, rows=1, cols=1)
    fig.add_traces(sigma_fig.data, rows=1, cols=2)
    fig.add_traces(theta_fig.data, rows=2, cols=1)
    fig.add_traces(delta_fig.data, rows=2, cols=2)

    fig.layout["coloraxis1"] = theta_fig.layout["coloraxis"]
    fig.layout["coloraxis1"]["colorbar_x"] = -0.1
    fig.layout["coloraxis1"]["colorbar_y"] = 0.4
    fig.layout["coloraxis1"]["colorbar_len"] = 0.8
    fig.layout["coloraxis2"] = delta_fig.layout["coloraxis"]
    fig.layout["coloraxis2"]["colorbar_x"] = 1.02
    fig.layout["coloraxis2"]["colorbar_y"] = 0.4
    fig.layout["coloraxis2"]["colorbar_len"] = 0.8
    
    fig.data[2]["coloraxis"] = "coloraxis1"
    fig.data[3]["coloraxis"] = "coloraxis2"

    print(fig)
   
    return fig 


if __name__=="__main__":

    model_hdf = sys.argv[1]
    out_html = sys.argv[2]     

    cancer_types = su.load_sample_groups(model_hdf)

    ################################
    # THETA PLOT
    feature_batch_ids,\
    sample_batch_ids, \
    theta_value_dicts,\
    internal_sample_idx = su.load_batch_matrix(model_hdf, "matfac", 
                                               "matfac/theta_values", 
                                               keytype=str, dtype=float)
    print(feature_batch_ids)
    batch_df, theta_df = make_batch_dfs(feature_batch_ids,
                                        sample_batch_ids,
                                        theta_value_dicts,
                                        internal_sample_idx)
    
    theta_fig = make_batch_heatmap(batch_df, theta_df, cancer_types)
 
    ################################
    # LOG DELTA PLOT 
    feature_batch_ids,\
    sample_batch_ids,\
    log_delta_value_dicts,\
    internal_sample_idx = su.load_batch_matrix(model_hdf, "matfac", 
                                                 "matfac/log_delta_values", 
                                                 keytype=str, dtype=float)
    
    batch_df, log_delta_df = make_batch_dfs(feature_batch_ids,
                                            sample_batch_ids,
                                            log_delta_value_dicts,
                                            internal_sample_idx)
    
    delta_df = log_delta_df.apply(np.exp)
    delta_fig = make_batch_heatmap(batch_df, delta_df, cancer_types, 
                                   color_scale="puor_r", zmin=0.0, zmax=2.0)

    ################################
    # Mu plot

    features = su.load_features(model_hdf)
    mu = su.load_mu(model_hdf)
  
    mu_fig = col_param_plot(mu, features)


    ################################
    # Sigma plot
    log_sigma = su.load_log_sigma(model_hdf)
    sigma_fig = col_param_plot(np.exp(log_sigma), features)


    ################################
    # Combine figures
    fig = combine_figs(theta_fig, delta_fig,
                       mu_fig, sigma_fig)

    fig.write_html(out_html)

