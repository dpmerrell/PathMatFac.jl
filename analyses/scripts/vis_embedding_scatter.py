
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
                         CategoricalColorMapper, WheelZoomTool
from bokeh.transform import transform

import script_util as su

def embedding_scatter_interactive(X, instance_ids, instance_groups):

    u, s, vh = np.linalg.svd(X, full_matrices=False)

    pcs = u * s

    x = pcs[:,0]
    y = pcs[:,1]
    group_encoder = {gp:idx for idx, gp in enumerate(np.unique(instance_groups))}
    colors = np.vectorize(lambda x: Category20[20][group_encoder[x]%20])(instance_groups)

    source = ColumnDataSource(data=dict(x=x, y=y, desc=instance_groups, 
                                                  pat=instance_ids, 
                                                  color=colors))

    hover = HoverTool(tooltips=[('Patient', '@pat'), ('Group', '@desc')])
    mapper = CategoricalColorMapper(factors=np.unique(colors), palette=Category20[20])

    p = figure(plot_width=800, plot_height=800, tools=[hover, BoxZoomTool(), WheelZoomTool()], title="Pathway Embedding")
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
    input_hdf = args[1]
    output_scatter = args[2]

    orig_samples,\
    orig_groups, \
    aug_samples, \
    sample_to_idx = su.load_sample_info(input_hdf)

    orig_sample_idx = np.vectorize(lambda x: sample_to_idx[x])(orig_samples)

    original_genes,\
    original_assays,\
    augmented_genes, feat_to_idx = su.load_feature_info(input_hdf)
    
    X = su.load_embedding(input_hdf)
    orig_X = X[orig_sample_idx,:]

    #plt.rcParams['legend.fontsize'] = 'x-small'

    #embedding_scatter(orig_X, orig_groups)
    #plt.savefig(output_scatter, dpi=400)

    p = embedding_scatter_interactive(orig_X, orig_samples, orig_groups) 
    output_file(output_scatter)
    show(p)
