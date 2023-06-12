
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


def factor_line_plot(Y, factor_idx=[0,1,2,3,5,12], feature_groups=None, 
                                            w=6, h=8, group_label_rotation=0.0,
                                            group_label_size=6):
   
    N = Y.shape[1]

    nfactors = len(factor_idx) 
    height_ratios = [0.98/nfactors]*nfactors + [0.02]
    f, axs = plt.subplots(nrows=nfactors+1, ncols=1, sharex=True, figsize=(w,h), gridspec_kw={"height_ratios":height_ratios})

    for i in range(nfactors):
        k = factor_idx[i]
        y = Y[k,:]
        axs[i].plot(range(1,N+1), y, color="k", linewidth=0.75)
        axs[i].set_ylabel(f"Factor {k+1}")
        axs[i].set_xlim([1,N+1])
        axs[i].set_xticks([])
        axs[i].set_xticklabels([])
        ylim = max(5, np.max(np.abs(y))*1.05)
        axs[i].set_ylim([-ylim, ylim])
        axs[i].set_xticks([-ylim, ylim])
        axs[i].set_xticklabels([-ylim, ylim])
        axs[i].plot(range(1,N+1), np.zeros(N), "--", color="k", linewidth=0.5)
 
    flags, xticks, labels, midpoints = labels_to_indicators(feature_groups)
    
    axs[nfactors].imshow(flags.reshape(1,N), aspect="auto", cmap="binary", vmin=0, vmax=1, interpolation="none")
    axs[nfactors].set_xticks(xticks)
    axs[nfactors].set_xticklabels(xticks)
    axs[nfactors].set_yticks([])
    axs[nfactors].set_yticklabels([])
    axs[nfactors].set_xlabel("Columns")

    for label, midpoint in zip(labels, midpoints):
        axs[nfactors].text(midpoint, 1.5, label, rotation=group_label_rotation, size=group_label_size, 
                             horizontalalignment="center", verticalalignment="top")
    plt.suptitle("Linear factors")

    return f


def plot_factors(in_hdf):

    fig = None
    with h5py.File(in_hdf, "r") as hfile:
        Y = hfile["Y"][:,:].transpose() 
        feature_groups = hfile["feature_views"][:].astype(str)
        fig = factor_line_plot(Y, feature_groups=feature_groups)

    return fig


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("params_hdf")
    parser.add_argument("out_png")
    parser.add_argument("--dpi", type=int, default=300)

    args = parser.parse_args()

    in_hdf = args.params_hdf
    out_png = args.out_png
    dpi = args.dpi

    f = plot_factors(in_hdf)
    f.tight_layout(h_pad=0.05, w_pad=0.05)

    f.savefig(out_png, dpi=dpi)

