
import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys
import os
import bokeh
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.palettes import plasma, Category20
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, BoxZoomTool,\
                         CategoricalColorMapper, WheelZoomTool, PanTool
from bokeh.transform import transform

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


def embedding_scatter_interactive(X, instance_ids, instance_groups, clinical_data, clinical_cols):

    u, s, vh = np.linalg.svd(X, full_matrices=False)

    pcs = u * s

    x = pcs[:,0]
    y = pcs[:,1]
    group_encoder = {gp:idx for idx, gp in enumerate(np.unique(instance_groups))}
    colors = np.vectorize(lambda x: Category20[20][group_encoder[x]%20])(instance_groups)

    source_dict = dict(x=x, y=y, desc=instance_groups, pat=instance_ids, color=colors)
    for i, col in enumerate(clinical_cols):
        source_dict[col] = clinical_data[:,i] 
    source = ColumnDataSource(data=source_dict)

    nice_clinical_cols = [su.NICE_NAMES[col] for col in clinical_cols]
    tooltips = [('Patient', '@pat'), ('Cancer', '@desc')]
    for nice_col, col in zip(nice_clinical_cols, clinical_cols):
        tooltips.append((nice_col, "@"+col))

    hover = HoverTool(tooltips=tooltips)
    mapper = CategoricalColorMapper(factors=np.unique(colors), palette=Category20[20])

    p = figure(plot_width=800, plot_height=800, tools=[hover, BoxZoomTool(), PanTool()], title="Pathway Embedding")
    p.circle('x', 'y', size=8, source=source, fill_color='color', fill_alpha=0.9, line_alpha=0.0)

    return p 


def embedding_scatter(X, instance_groups):


    u, s, vh = np.linalg.svd(X, full_matrices=False)

    pcs = u * s

    x = pcs[:,0]
    y = pcs[:,1]
    #group_encoder = {gp:idx for idx, gp in enumerate(np.unique(instance_groups))}
    group_idx = {gp: [] for gp in np.unique(instance_groups)}
    for i, gp in enumerate(instance_groups):
        group_idx[gp].append(i)
    group_idx = {gp: np.array(idx_ls) for (gp, idx_ls) in group_idx.items()}

    markers = ["o", "*", "+"]

    for (i, (gp, idx)) in enumerate(group_idx.items()):
        print(gp, idx.shape, i)
        plt.scatter(x[idx],y[idx], s=2.0, c=[float(i%20) for _ in idx], 
                    marker=markers[i%3], cmap="tab20", label=gp,
                    vmin=0, vmax=20)

    plt.legend(ncol=2)

    return


if __name__=="__main__":

    args = sys.argv
    model_hdf = args[1]
    clinical_hdf = args[2]
    output_scatter = args[3]

    orig_samples,\
    orig_groups, \
    aug_samples, \
    sample_to_idx = su.load_sample_info(model_hdf)

    orig_sample_idx = np.vectorize(lambda x: sample_to_idx[x])(orig_samples)

    original_genes,\
    original_assays,\
    augmented_genes,\
    augmented_assays, feat_to_idx = su.load_feature_info(model_hdf)
    
    X = su.load_embedding(model_hdf)
    orig_X = X[orig_sample_idx,:]

    clinical_cols = ["gender", "hpv_status"] #, "age_at_pathologic_diagnosis", "tobacco_smoking_history", "race"] 

    clinical_data, clinical_samples = load_clinical_data(clinical_hdf, clinical_cols)
    print("CLINICAL_DATA:")
    print(clinical_data)
    clinical_data = match_clinical_to_omic(clinical_data, clinical_samples, orig_samples)

    print("NEW_CLINICAL_DATA:")
    print(clinical_data)
    p = embedding_scatter_interactive(orig_X, orig_samples, orig_groups, clinical_data, clinical_cols) 
    output_file(output_scatter)
    show(p)


