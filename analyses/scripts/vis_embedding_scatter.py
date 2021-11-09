
import numpy as np
import h5py
import sys
import os

import chart_studio.plotly as cs
import plotly.express as px
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


def embedding_scatter_plotly(X, instance_ids, instance_groups, clinical_data, clinical_cols, pc_idx):

    u, s, vh = np.linalg.svd(X, full_matrices=False)

    pcs = u * s

    x = pcs[:,pc_idx[0]]
    y = pcs[:,pc_idx[1]]
    z = pcs[:,pc_idx[2]]

    labels = ["PC {}".format(idx+1) for idx in pc_idx]

    source_df = pd.DataFrame({
                              labels[0]: x, 
                              labels[1]: y, 
                              labels[2]: z, 
                              "Cancer Type": instance_groups, 
                              "Patient ID": instance_ids
                             }
                            )

    hover_cols = []

    for i, col in enumerate(clinical_cols):
        col_name = su.NICE_NAMES[col]
        source_df[col_name] = clinical_data[:,i]
        hover_cols.append(col_name)

    fig = px.scatter_3d(source_df, x=labels[0], y=labels[1], z=labels[2],
                                   color="Cancer Type",
                                   hover_data=["Cancer Type"]+hover_cols)

    fig.update_layout(scene={"xaxis":{"range":[x.min(), x.max()]},
                             "yaxis":{"range":[y.min(), y.max()]},
                             "zaxis":{"range":[z.min(), z.max()]},
                            }
                     )

    return fig 



if __name__=="__main__":

    args = sys.argv
    model_hdf = args[1]
    clinical_hdf = args[2]
    first_pc = int(args[3])
    output_scatter = args[4]

    orig_samples,\
    orig_groups, \
    aug_samples, \
    sample_to_idx = su.load_sample_info(model_hdf)

    orig_sample_idx = np.vectorize(lambda x: sample_to_idx[x])(orig_samples)
    
    if len(args) == 6:
        groups_to_plot = set(args[5].split(":"))
    else:
        groups_to_plot = set(orig_groups) 
    
    kept_idx = np.vectorize(lambda x: x in groups_to_plot)(orig_groups)
    orig_samples = orig_samples[kept_idx]
    orig_groups = orig_groups[kept_idx]
    orig_sample_idx = orig_sample_idx[kept_idx]

    original_genes,\
    original_assays,\
    augmented_genes,\
    augmented_assays, feat_to_idx = su.load_feature_info(model_hdf)
    
    X = su.load_embedding(model_hdf)
    orig_X = X[orig_sample_idx,:]

    clinical_cols = ["gender", "hpv_status"] #, "age_at_pathologic_diagnosis", "tobacco_smoking_history", "race"] 

    clinical_data, clinical_samples = load_clinical_data(clinical_hdf, clinical_cols)
    clinical_data = match_clinical_to_omic(clinical_data, clinical_samples, orig_samples)

    pcs = list(range(first_pc, first_pc+3))

    fig = embedding_scatter_plotly(orig_X, orig_samples, orig_groups, clinical_data, clinical_cols, pcs)
    fig.write_html(output_scatter) 

