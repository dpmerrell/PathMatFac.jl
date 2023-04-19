

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

import vis_model_params as vmp


def plot_factor_comparison(Y1, Y2, feature_groups, suptitle, title1, title2, w=6, h=4, rotate=True):

    if rotate:
        print("Rotating Y2 into alignment with Y1")
        u1, s1, v1_t = np.linalg.svd(Y1, full_matrices=False)
        u2, s2, v2_t = np.linalg.svd(Y2, full_matrices=False)
        Y2 = ((u1 @ v1_t) @ (u2 @ v2_t).transpose()) @ Y2

    f, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(w,h), gridspec_kw={"height_ratios":[0.95,0.05]
                                                                                     "width_ratios":[0.5,0.5]})

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
    plt.suptitle(suptitle)

    f.tight_layout(h_pad=0.05, w_pad=0.05)

    return f



if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("params_hdf_1")
    parser.add_argument("params_hdf_2")
    parser.add_argument("out_png")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--suptitle", type=str, default="Comparison of factors")
    parser.add_argument("--title_1", type=str, default="True factors")
    parser.add_argument("--title_2", type=str, default="Fitted factors")

    args = parser.parse_args()

    params_hdf_1 = args.params_hdf_1
    params_hdf_2 = args.params_hdf_2
    out_png = args.out_png
    dpi = args.dpi
    suptitle = args.suptitle
    title_1 = args.title_1
    title_2 = args.title_2

    f = plot_factor_comparison(in_hdf, target_param=target_param)

    f.savefig(out_png, dpi=dpi)


if __name__=="__main__":

    
