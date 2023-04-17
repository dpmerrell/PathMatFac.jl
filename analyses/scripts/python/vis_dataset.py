
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from copy import copy
import script_util as su
import numpy as np
import argparse
import h5py
import vis_model_params as vmp

mpl.rcParams['font.family'] = "serif"

NAMES = su.NICE_NAMES
ASSAY_ORDER = ["mutation", "methylation", "mrnaseq", "rppa", "cna"]

ASSAY_CMAPS = {"mutation": {"vmin": 0,
                            "vmax": 1,
                            "cmap": "binary"},
               "methylation": {"vmin": -3.0,
                               "vmax": 3.0,
                               "cmap": "bwr_r"},
               "rppa": {"vmin": -2.0,
                        "vmax": 2.0,
                        "cmap": "bwr_r"},
               "mrnaseq": {"vmin": -1.0,
                           "vmax": 15.0,
                           "cmap": "inferno_r"},
               "cna": {"vmin": 1.0,
                       "vmax": 3.0,
                       "cmap": "bwr_r"}
               }


def data_heatmap(mat, name, ax, vmin=-2.0, vmax=2.0, origin="upper", cmap="bwr_r"):
   
    print("\tmax: ", np.nanmax(mat))
    print("\tmin: ", np.nanmin(mat)) 
    print("\tmean: ", np.nanmean(mat))
    print("\tstd: ", np.nanstd(mat))
    (M, N) = mat.shape
    cm = copy(getattr(plt.cm, cmap)) 
    cm.set_bad(color=(0.5, 0.5, 0.5))

    ax.imshow(mat, aspect="auto", cmap=cm, vmin=vmin, vmax=vmax, origin=origin,
                   interpolation="none")
    #ax.set_xticks([0, N-1], [1, N])

    ax.set_title(f"{NAMES[name]}\n({N})", size=8) 
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])


def plot_data(in_hdf, w=6, h=5, suptitle="Multiomic dataset", ylabel="Cancer types", group_label_size=6):

    fig = None
    with h5py.File(in_hdf, "r") as f:
        feature_assays = f["omic_data/feature_assays"][:].astype(str)
        
        unq_assays = np.unique(feature_assays)
        print(unq_assays)
        used_assays = [a for a in ASSAY_ORDER if a in unq_assays]

        # Figure out how many features belong to each assay
        assay_N = []
        a_idx = [] 
        for ua in used_assays:
            a_idx.append(np.where(feature_assays == ua)[0])
            assay_N.append(len(a_idx[-1]))
       
        # Set up the subplots
        width_ratios = [0.025*np.sum(assay_N)] + assay_N 
        fig, axs = plt.subplots(nrows=2, ncols=len(used_assays)+1, #sharey=True, 
                                         gridspec_kw={"height_ratios": [0.975, 0.025],
                                                      "width_ratios": width_ratios},
                                         figsize=(w,h))

        #all_data = f["omic_data/data"][:,:].transpose()
        N,M = f["omic_data/data"].shape
        print(M,N)
        for i, ua in enumerate(used_assays):
            print(ua)
            a_data = f["omic_data/data"][a_idx[i],:].transpose()
            vmin = ASSAY_CMAPS[ua]["vmin"]
            vmax = ASSAY_CMAPS[ua]["vmax"]
            data_heatmap(a_data, ua, axs[0][i+1], vmin=vmin, vmax=vmax,
                                                  cmap=ASSAY_CMAPS[ua]["cmap"])
            sm = ScalarMappable(cmap=ASSAY_CMAPS[ua]["cmap"])
            sm.set_clim(vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(mappable=sm, ticks=[vmin, vmax], cax=axs[1][i+1], fraction=1.0, orientation="horizontal")
            #axs[1][i+1].set_yticks([])
            #axs[1][i+1].set_yticklabels([])
            cbar.set_ticks([vmin, vmax], ["{:.0f}".format(vmin), "{:.0f}".format(vmax)])
            #axs[1][i+1].set_xticklabels()

        axs[1][0].set_visible(False)

        sample_conditions = f["omic_data/instance_groups"][:].astype(str)
        flags, ticks, unq_labels, midpoints = vmp.labels_to_indicators(sample_conditions)
        counts = np.array(ticks[1:]) - np.array(ticks[:-1])
        axs[0][0].imshow(flags.reshape(M,1), aspect="auto", cmap="binary", vmin=0, vmax=1, origin="upper", interpolation="none")
        axs[0][0].set_yticks([ticks[-1]])
        axs[0][0].set_yticklabels([ticks[-1]], fontsize=6)
        axs[0][0].set_ylabel(ylabel)
        axs[0][0].set_xticks([])
        axs[0][0].set_xticklabels([])
        for label, midpoint, c, in zip(unq_labels, midpoints, counts):
            axs[0][0].text(-1.0, midpoint, f"{label}\n({c})", rotation=0.0, size=group_label_size, 
                                 horizontalalignment="right", verticalalignment="center")

    plt.suptitle(suptitle) 
    fig.tight_layout(h_pad=0.01, w_pad=0.01)
     
    return fig
             


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("data_hdf")
    parser.add_argument("out_png")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--title", type=str, default="Multiomic dataset")
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--height", type=int, default=6)

    args = parser.parse_args()

    in_hdf = args.data_hdf
    out_png = args.out_png
    dpi = args.dpi
    suptitle = args.title

    print("TITLE: ", suptitle)

    f = plot_data(in_hdf, suptitle=suptitle, w=args.width, h=args.height)

    f.savefig(out_png, dpi=dpi)

