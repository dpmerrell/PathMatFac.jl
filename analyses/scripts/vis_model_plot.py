import numpy as np
import matplotlib.pyplot as plt
import sys
import script_util as su


def plot_heatmap(matrix):
    plt.matshow(matrix, cmap="bwr", vmin=-0.5, vmax=0.5)


def plot_instance_groups(gp_vec):

    # Get the unique group IDs
    # (in a way that preserves order)
    unq_gps = []
    unq_gp_set = {}
    gp_start_idx = [] # Also get the start idx for each group
    for idx, gp in enumerate(gp_vec):
        if gp not in unq_gp_set:
            unq_gp_set.add(gp)
            unq_gps.append(gp)
            gp_start_idx.append(idx)

    # Translate groups to integers; and then parity
    gp_encoder = {gp: idx for (idx, gp) in enumerate(unq_gps)}
    gp_colors = np.vectorize(lambda x: gp_encoder[x] % 2)(gp_vec)

    gp_color_mat = np.reshape(gp_colors, (len(gp_colors), 1))

    plt.matshow(gp_color_mat, cmap="binary")



if __name__=="__main__":

    args = sys.argv
    model_hdf = args[1]
    
    original_samples, \
    original_groups, \
    augmented_samples, \
    sample_to_idx = load_sample_info(model_hdf) 

    orig_sample_idx = np.vectorize(lambda x: sample_to_idx[x])(original_samples)

    original_genes, \
    original_assays, \
    augmented_genes, \
    feat_to_idx = load_feature_info(model_hdf)

    orig_feat_idx = np.array([feat_to_idx[pair] for pair in zip(original_genes, original_assays)])

    #aug_data = su.load_data(model_hdf)

    #orig_data = 
