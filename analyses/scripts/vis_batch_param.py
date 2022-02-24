

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


def col_param_plot(param_df):
    fig = px.line(param_df)

    return fig


#def restrict_idx_array(idx_array, restriction):
#    
#    restriction_set = set(restriction)
#    return [idx for idx in idx_array if idx not in restriction_set]
#
#
#def restrict_idx_dict(idx_dict, restriction):
#    return {k: restrict_idx_array(arr, restriction) for k, arr in idx_dict.items()}


def make_batch_dfs(feature_batch_ids, sample_batch_ids, value_dicts, internal_sample_idx):

    valid_fb_idx = [idx for idx, fb in enumerate(feature_batch_ids) if fb != ""]
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


def combine_figs(theta_fig, delta_fig):

    fig = make_subplots(rows=1,cols=2,
                        subplot_titles=["Batch Shift",
                                        "Batch Scale"],
                        ) 
    fig.add_traces(theta_fig.data, rows=1, cols=1)
    fig.add_traces(delta_fig.data, rows=1, cols=2)

    fig.layout["coloraxis1"] = theta_fig.layout["coloraxis"]
    fig.layout["coloraxis1"]["colorbar_x"] = -0.1
    fig.layout["coloraxis2"] = delta_fig.layout["coloraxis"]
    fig.layout["coloraxis2"]["colorbar_x"] = 1.02
    
    #fig.layout["coloraxis2"]["colorbar"]["tickvals"] = [-3,-2,-1,0,1,2,3]
    #fig.layout["coloraxis2"]["colorbar"]["ticktext"] = ["0.125","0.25",
    #                                                    "0.5", "1.0",
    #                                                    "2.0", "4.0", "8.0"]

    fig.data[0]["coloraxis"] = "coloraxis1"
    fig.data[1]["coloraxis"] = "coloraxis2"
   
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

    batch_df, theta_df = make_batch_dfs(feature_batch_ids,
                                        sample_batch_ids,
                                        theta_value_dicts,
                                        internal_sample_idx)
    
    theta_fig = make_batch_heatmap(batch_df, theta_df, cancer_types)
    print(theta_fig)
 
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
    #delta_df = log_delta_df.apply(np.log2)
    delta_fig = make_batch_heatmap(batch_df, delta_df, cancer_types, 
                                   color_scale="puor_r", zmin=0.0, zmax=2.0)

    fig = combine_figs(theta_fig, delta_fig)

    print(fig)

    fig.write_html(out_html)

