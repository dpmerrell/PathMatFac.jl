import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys

import script_util as su


def plot_heatmap(ax, matrix, xlabel=None, ylabel=None, vmin=-1.0, vmax=1.0):

    cmap = matplotlib.cm.bwr
    cmap.set_bad("gray",1.0)

    ax.matshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def group_bar_encoder(gp_vec):
    
    # Get the unique group IDs
    # (in a way that preserves order)
    unq_gps = []
    unq_gp_set = set([])
    gp_start_idx = [] # Also get the start idx for each group
    for idx, gp in enumerate(gp_vec):
        if gp not in unq_gp_set:
            unq_gp_set.add(gp)
            unq_gps.append(gp)
            gp_start_idx.append(idx)

    # Translate groups to integers; and then parity
    gp_encoder = {gp: idx for (idx, gp) in enumerate(unq_gps)}
    gp_colors = np.vectorize(lambda x: gp_encoder[x] % 2)(gp_vec)

    gp_end_idx = gp_start_idx[1:] + [len(gp_vec)]
    gp_loc_idx = [0.5*(pair[0] + pair[1]) for pair in zip(gp_start_idx, gp_end_idx)]

    return gp_colors, unq_gps, gp_loc_idx


def plot_instance_groups(ax, gp_vec):

    gp_colors, gp_names, gp_loc_idx = group_bar_encoder(gp_vec)

    gp_color_mat = np.reshape(gp_colors, (len(gp_colors), 1))

    ax.matshow(gp_color_mat, cmap="binary", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_yticks(gp_loc_idx)
    ax.set_yticklabels(gp_names, rotation=45)
    ax.set_xticks([])
    return 


def plot_feature_assays(ax, assay_vec):

    assay_colors, assay_names, assay_loc_idx = group_bar_encoder(assay_vec)

    assay_color_mat = np.reshape(assay_colors, (1,len(assay_colors)))

    ax.matshow(assay_color_mat, cmap="binary", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(assay_loc_idx)
    ax.set_xticklabels(assay_names)
    ax.set_yticks([])

    return



def plot_model(gp_vec, assay_vec, X, Y, sample_offset, feature_offset, omic_values,
               fig_width=20.0, title="Fitted Model"):

    M = len(gp_vec)
    N = len(assay_vec)
    K = X.shape[0]
    aspect_ratio = M/N
    fig_height = aspect_ratio * fig_width

    fig = plt.figure(figsize=(fig_width,fig_height))

    gs = gridspec.GridSpec(nrows=4, ncols=4, 
                           height_ratios=[K, 3.0*K, K, M], 
                           width_ratios=[K, 3.0*K, K, N])
    fig.subplots_adjust(wspace=0.0, hspace=0.0)


    # Feature assay indicator
    ax1 = fig.add_subplot(gs[0,3])
    plot_feature_assays(ax1, assay_vec)
    ax1.tick_params(top=True)

    # Y factor
    ax2 = fig.add_subplot(gs[1,3])
    plot_heatmap(ax2, Y, vmin=-2.0, vmax=2.0)

    # feature offset
    ax3 = fig.add_subplot(gs[2,3])
    plot_heatmap(ax3, np.reshape(feature_offset, (1,len(feature_offset))))

    # group indicator
    ax4 = fig.add_subplot(gs[3,0])
    plot_instance_groups(ax4, gp_vec)

    # X factor
    ax5 = fig.add_subplot(gs[3,1])
    plot_heatmap(ax5, np.transpose(X), vmin=-0.02, vmax=0.02)

    # instance offset
    ax6 = fig.add_subplot(gs[3,2])
    plot_heatmap(ax6, np.reshape(sample_offset, (len(sample_offset),1)))

    # Omic values
    ax7 = fig.add_subplot(gs[3,3])
    plot_heatmap(ax7, omic_values, vmin=-5.0, vmax=5.0)

    plt.suptitle(title)

    return


if __name__=="__main__":

    args = sys.argv
    model_hdf = args[1]
    aug_omic_hdf = args[2]
    output_png = args[3]

    orig_samples, \
    orig_groups, \
    augmented_samples, \
    sample_to_idx = su.load_sample_info(model_hdf) 

    orig_sample_idx, aug_sample_idx = su.keymatch(orig_samples, augmented_samples)

    orig_genes, \
    orig_assays, \
    augmented_genes, \
    augmented_assays, \
    aug_feat_to_idx = su.load_feature_info(model_hdf)

    orig_features = list(zip(orig_genes, orig_assays))
    aug_features = list(zip(augmented_genes, augmented_assays))
    orig_feat_idx, aug_feat_idx = su.keymatch(orig_features, aug_features)

    X = su.load_hdf(model_hdf, "/matfac/X").transpose()
    Y = su.load_hdf(model_hdf, "/matfac/Y").transpose()

    orig_X = X[:,aug_sample_idx]
    orig_Y = Y[:,aug_feat_idx]
   
    omic_values = su.load_hdf(aug_omic_hdf, "/omic_matrix").transpose()
    #print("OMIC VALUES:", omic_values.shape)
    omic_values = omic_values[aug_sample_idx,:]
    omic_values = omic_values[:,aug_feat_idx] 

    sample_offset = su.load_hdf(model_hdf, "/matfac/instance_offset")[aug_sample_idx]
    feature_offset = su.load_hdf(model_hdf, "/matfac/feature_offset")[aug_feat_idx]

    #print("ORIG X:", orig_X.shape)
    #print("ORIG Y:", orig_Y.shape)
    #print("OMIC VALUES:", omic_values.shape)
    #print("SAMPLE_OFFSET:", sample_offset.shape)
    #print("FEATURE_OFFSET:", feature_offset.shape)

    #print("ORIG_GROUPS:", orig_groups.shape)
    orig_assays = orig_assays[orig_feat_idx]
    #print("ORIG_ASSAYS:", orig_assays.shape)

    plot_model(orig_groups, orig_assays,
               orig_X, orig_Y, 
               sample_offset, feature_offset,
               omic_values)

    plt.tight_layout(h_pad=1.0)

    plt.savefig(output_png, dpi=400)



