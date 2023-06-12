
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
import script_util as su
import numpy as np
import argparse
import h5py

mpl.rcParams['font.family'] = "serif"

NAMES = su.NICE_NAMES


def plot_scree(in_hdf, w=5.0, h=3.0):

    fig = None
    Y = None
    assays = None
    with h5py.File(in_hdf, "r") as hfile:
        Y = hfile["Y"][:,:].transpose()
        assays = hfile["feature_views"][:].astype(str)

    print(Y.shape)
    fig = plt.figure(figsize=(w,h))

    k = np.arange(1, Y.shape[0]+1)
    Y_sqnorm = np.sum(Y*Y, axis=1)
    plt.plot(k, Y_sqnorm, linewidth=1.0, color="k", label="Total")
 
    unq_assays = np.unique(assays)
    for (i,a) in enumerate(unq_assays):
        rel_Y = Y[:,assays == a]
        rel_sqnorm = np.sum(rel_Y*rel_Y, axis=1)
        plt.plot(k, rel_sqnorm, "--", linewidth=0.675, color=su.ALL_COLORS[i], label=a)

    plt.title("Multiomic scree plot")
    plt.xlabel("Linear factor")
    plt.ylabel("Squared norm")

    plt.xlim([1,Y.shape[0]])
    plt.ylim([0, np.max(Y_sqnorm)*1.1])
    plt.yticks([0, np.max(Y_sqnorm)])

    plt.legend()

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

    f = plot_scree(in_hdf)
    f.tight_layout(h_pad=0.05, w_pad=0.05, pad=0.5)
    f.savefig(out_png, dpi=dpi)

