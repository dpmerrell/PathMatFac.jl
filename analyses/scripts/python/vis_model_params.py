
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
import script_util as su
import numpy as np
import argparse
import h5py

mpl.rcParams['font.family'] = "serif"

NAMES = su.NICE_NAMES


def matrix_heatmap(mat, x_groups=None, dpi=300, w=6, h=3, cmap="Greys", vmin=-2.0, vmax=2.0,
                        xlabel="Features", ylabel="Factors", origin="lower",
                        title="Matrix $Y$"):

    (K, N) = mat.shape

    f = plt.figure(figsize=(w, h))

    sm = ScalarMappable(cmap=cmap)
    sm.set_clim(vmin=vmin, vmax=vmax)

    img = plt.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin=origin,
                          interpolation="none")

    cbar = plt.colorbar(mappable=sm, fraction=0.05, ticks=[vmin, vmax])

    plt.title(title) 
    plt.yticks([0, K-1], [1, K])
    plt.xticks([0, N-1], [1, N])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    return f


def col_param_plot(mu, logsigma, w=6, h=4):
   
    N = len(mu)
 
    #f = plt.figure(figsize=(w,h))
    f, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(w,h))

    axs[0].plot(range(1,N+1), mu, color="k", linewidth=0.5)
    axs[0].set_ylabel("Column shift")
    
    axs[1].plot(range(1,N+1), np.exp(logsigma), color="k", linewidth=0.5)
    axs[1].set_ylabel("Column scale")
    axs[1].set_xlabel("Columns")

    plt.xlim([1,N+1])
    plt.suptitle("Column parameters")

    return f


def plot_param(in_hdf, target_param="Y"):

    fig = None
    with h5py.File(in_hdf, "r") as hfile:

        if target_param == "Y":
            Y = hfile["Y"][:,:].transpose() 
            fig = matrix_heatmap(Y)
        elif target_param == "X":
            X = hfile["X"][:,:].transpose()
            fig = matrix_heatmap(X, title="Matrix $X$", ylabel="Embedding dims.",
                                                      xlabel="Samples")
        elif target_param == "col_params":
            mu = hfile["mu"][:]
            logsigma = hfile["logsigma"][:]
            fig = col_param_plot(mu, logsigma)

    return fig


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("params_hdf")
    parser.add_argument("out_png")
    parser.add_argument("--param", choices=["X", "Y", "col_params", "delta", "theta", "S", "A"], default="Y")
    parser.add_argument("--dpi", type=int, default=300)

    args = parser.parse_args()

    in_hdf = args.params_hdf
    out_png = args.out_png
    target_param = args.param
    dpi = args.dpi

    f = plot_param(in_hdf, target_param=target_param)

    f.tight_layout()
    f.savefig(out_png, dpi=dpi)

