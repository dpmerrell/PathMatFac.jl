
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
import script_util as su
import numpy as np
import argparse
import h5py

mpl.rcParams['font.family'] = "serif"
#plt.rcParams.update({'text.usetex': True,
#                     'font.family': 'serif'})

NAMES = su.NICE_NAMES


"""
    Given an array of labels occurring in contiguous blocks,
    return an array of zeros and ones indicating the contiguous blocks.
    Also return the unique labels and the midpoints of their corresponding blocks.
"""
def labels_to_indicators(labels):
    
    _, unq_idx = np.unique(labels, return_index=True)
    unq_labels = labels[np.sort(unq_idx)]
    xticks = []
    midpoints = []
    indicators = np.zeros_like(labels, dtype=int)
    cur_ind = True

    for (i, ul) in enumerate(unq_labels):
        idx = np.where(labels == ul)[0]
        l_idx = idx[0]
        u_idx = idx[-1]
        m_idx = int((l_idx + u_idx)//2)
       
        if i == 0:
            xticks.append(l_idx)
        
        xticks.append(u_idx)
        midpoints.append(m_idx)

        indicators[l_idx:u_idx] = cur_ind
        cur_ind = (cur_ind + 1) % 2
       
    return indicators, xticks, unq_labels, midpoints 



def matrix_heatmap(mat, x_groups=None, dpi=300, w=6, h=3, cmap="Greys", vmin=-2.0, vmax=2.0,
                        xlabel="Features", ylabel="Factors", origin="lower",
                        title="Matrix Y", group_label_rotation=0.0, group_label_size=6):

    (K, N) = mat.shape
    f = None
    sm = ScalarMappable(cmap=cmap)
    sm.set_clim(vmin=vmin, vmax=vmax)

    if x_groups is None:
        f = plt.figure(figsize=(w, h))
        img = plt.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin=origin,
                              interpolation="none")
        plt.xticks([0, N-1], [1, N])
        cbar = plt.colorbar(mappable=sm, fraction=0.05, ticks=[vmin, vmax])

        plt.title(title) 
        plt.xlabel(xlabel)
        plt.yticks([0, K-1], [1, K])
        plt.ylabel(ylabel)

    else:
        f, axs = plt.subplots(nrows=2, ncols=2, sharex=True, gridspec_kw={"height_ratios":[0.95,0.05],
                                                                          "width_ratios": [0.95,0.05]},
                                                                          #"hspace": 0.05,
                                                                          #"wspace": 0.05}, 
                                                             figsize=(w,h))
        axs[0][0].imshow(mat, aspect="auto", cmap=cmap, 
                              vmin=vmin, vmax=vmax, origin=origin,
                              interpolation="none")
        axs[0][0].set_yticks([0, K-1])
        axs[0][0].set_yticklabels([1, K])
        axs[0][0].set_ylabel(ylabel)
        axs[0][0].set_xticks([])
        axs[0][0].set_xticklabels([])

        axs[0][1].set_visible(False)        
        cbar = plt.colorbar(mappable=sm, ticks=[vmin, vmax], ax=axs[0][1], fraction=1.0)

        flags, xticks, labels, midpoints = labels_to_indicators(x_groups)
        
        axs[1][0].imshow(flags.reshape(1,N), aspect="auto", cmap="binary", vmin=0, vmax=1, interpolation="none")
        axs[1][0].set_xticks(xticks)
        axs[1][0].set_xticklabels(xticks)
        axs[1][0].set_yticks([])
        axs[1][0].set_yticklabels([])
        axs[1][0].set_xlabel(xlabel)

        for label, midpoint in zip(labels, midpoints):
            axs[1][0].text(midpoint, 1.0, label, rotation=group_label_rotation, size=group_label_size, 
                                                 horizontalalignment="center", verticalalignment="top")

        axs[1][1].set_visible(False)

        plt.suptitle(title) 

    f.tight_layout(h_pad=0.05, w_pad=0.05)

    return f


def col_param_plot(mu, logsigma, feature_groups=None, w=6, h=4, group_label_rotation=0.0,
                                                                group_label_size=6):
   
    N = len(mu)

    if feature_groups is None: 
        f, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(w,h))

        axs[0].plot(range(1,N+1), mu, color="k", linewidth=0.5)
        axs[0].set_ylabel("Column shift")
        
        axs[1].plot(range(1,N+1), np.exp(logsigma), color="k", linewidth=0.5)
        axs[1].set_ylabel("Column scale")
        axs[1].set_xlabel("Columns")

        plt.xlim([1,N+1])
        plt.suptitle("Column parameters")

    else:
        f, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(w,h), gridspec_kw={"height_ratios":[0.49, 0.49,0.02]})
        axs[0].plot(range(1,N+1), mu, color="k", linewidth=0.75)
        axs[0].set_ylabel("Shift ($\mu$)")
        axs[0].set_xlim([1,N+1])
        axs[0].set_xticks([])
        axs[0].set_xticklabels([])
        
        axs[1].plot(range(1,N+1), np.exp(logsigma), color="k", linewidth=0.75)
        axs[1].set_ylabel("Scale ($\sigma$)")
        axs[1].set_xticks([])
        axs[1].set_xticklabels([])

        flags, xticks, labels, midpoints = labels_to_indicators(feature_groups)
        
        axs[2].imshow(flags.reshape(1,N), aspect="auto", cmap="binary", vmin=0, vmax=1, interpolation="none")
        axs[2].set_xticks(xticks)
        axs[2].set_xticklabels(xticks)
        axs[2].set_yticks([])
        axs[2].set_yticklabels([])
        axs[2].set_xlabel("Columns")

        for label, midpoint in zip(labels, midpoints):
            axs[2].text(midpoint, 1.5, label, rotation=group_label_rotation, size=group_label_size, 
                                              horizontalalignment="center", verticalalignment="top")
        plt.suptitle("Column parameters")

    f.tight_layout(h_pad=0.05, w_pad=0.05)

    return f


def batch_param_plot(values, col_ranges, ax, topk=100):
    for (v, cr) in zip(values, col_ranges):
        for i in range(min(v.shape[0], topk)):
            ax.plot(cr, v[i,:], linewidth=0.3)#, color="grey")
   

def plot_all_batch_params(hfile, w=6, h=5, group_label_rotation=0.0, group_label_size=6):

    f, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(w,h), 
                          gridspec_kw={"height_ratios":[0.49, 0.49, 0.02]})

    value_keys = sorted([k for k in list(hfile["theta"].keys()) if k.startswith("values_")])
    cr_keys = sorted([k for k in list(hfile["theta"].keys()) if k.startswith("col_range_")])

    shift_values = []
    shift_col_ranges = []
    for (vk, crk) in zip(value_keys, cr_keys):
        shift_values.append(hfile["theta"][vk][:,:].transpose())
        shift_col_ranges.append(hfile["theta"][crk][:].astype(int))

    batch_param_plot(shift_values, shift_col_ranges, axs[0])
    axs[0].set_ylabel("Shift ($\\theta$)")

    scale_values = []
    scale_col_ranges = []
    for (vk, crk) in zip(value_keys, cr_keys):
        scale_values.append(np.exp(hfile["logdelta"][vk][:,:].transpose()))
        scale_col_ranges.append(hfile["logdelta"][crk][:].astype(int))

    batch_param_plot(scale_values, scale_col_ranges, axs[1])
    axs[1].set_ylabel("Scale ($\delta$)")
    
    feature_groups = hfile["feature_views"][:].astype(str) 
    flags, xticks, labels, midpoints = labels_to_indicators(feature_groups)
    full_N = len(feature_groups)

    axs[2].imshow(flags.reshape(1,full_N), aspect="auto", cmap="binary", vmin=0, vmax=1, interpolation="none")
    axs[2].set_xticks(xticks)
    axs[2].set_xticklabels(xticks)
    axs[2].set_yticks([])
    axs[2].set_yticklabels([])
    axs[2].set_xlabel("Columns")

    for label, midpoint in zip(labels, midpoints):
        axs[2].text(midpoint, 1.5, label, rotation=group_label_rotation, size=group_label_size, 
                                          horizontalalignment="center", verticalalignment="top")

    axs[0].plot(range(full_N), np.zeros(full_N), linestyle="--", color="k", linewidth=1.0)
    axs[1].plot(range(full_N), np.ones(full_N), linestyle="--", color="k", linewidth=1.0)

    plt.suptitle("Batch effect parameters")

    f.tight_layout(h_pad=0.05, w_pad=0.05)

    return f


def plot_param(in_hdf, target_param="Y"):

    fig = None
    with h5py.File(in_hdf, "r") as hfile:
        if target_param == "Y":
            Y = hfile["Y"][:,:].transpose() 
            feature_groups = hfile["feature_views"][:].astype(str)
            fig = matrix_heatmap(Y, x_groups=feature_groups)
        elif target_param == "X":
            X = hfile["X"][:,:].transpose()
            sample_groups = hfile["sample_conditions"][:].astype(str)
            fig = matrix_heatmap(X, x_groups=sample_groups,
                                    title="Matrix X", 
                                    ylabel="Embedding dims.",
                                    xlabel="Samples")
        elif target_param == "col_params":
            mu = hfile["mu"][:]
            feature_groups = hfile["feature_views"][:].astype(str)
            logsigma = hfile["logsigma"][:]
            fig = col_param_plot(mu, logsigma, feature_groups=feature_groups)

        elif target_param == "batch_params":
            fig = plot_all_batch_params(hfile)

    return fig


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("params_hdf")
    parser.add_argument("out_png")
    parser.add_argument("--param", choices=["X", "Y", "col_params", "batch_params", "S", "A"], default="Y")
    parser.add_argument("--dpi", type=int, default=300)

    args = parser.parse_args()

    in_hdf = args.params_hdf
    out_png = args.out_png
    target_param = args.param
    dpi = args.dpi

    f = plot_param(in_hdf, target_param=target_param)

    f.savefig(out_png, dpi=dpi)

