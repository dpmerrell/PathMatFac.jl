

from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
import script_util as su
import numpy as np
import argparse
import h5py

mpl.rcParams['font.family'] = "serif"

NAMES = su.NICE_NAMES


def load_values(hfile, batch_param_name):
    value_keys = sorted([k for k in list(hfile[batch_param_name].keys()) if k.startswith("values_")])
    values = []
    for vk in value_keys:
        values.append(hfile[batch_param_name][vk][:,:].transpose())
    return values

def load_row_batch_ids(hfile, batch_param_name):
    rid_keys = sorted([k for k in list(hfile[batch_param_name].keys()) if k.startswith("batch_ids")])
    rbids = []
    for rk in rid_keys:
        rbids.append(hfile[batch_param_name][rk][:].astype(str))
    return rbids

def scatter_values(ax, true_v, fitted_v):
    #t_v = true_v.flatten()
    #f_v = fitted_v.flatten()
    #ax.scatter(t_v, f_v, color="k", s=0.5)
    for i in range(true_v.shape[0]):
        ax.scatter(true_v[i,:], fitted_v[i,:], s=0.5)


def compare_batch_params(true_file, fitted_file, batch_view_names, w=6, h=5):

    true_theta = load_values(true_file, "theta")
    fitted_theta = load_values(fitted_file, "theta")
   
    true_logdelta = load_values(true_file, "logdelta")
    fitted_logdelta = load_values(fitted_file, "logdelta")

    row_batch_ids = load_row_batch_ids(true_file, "theta")

    n_views = len(true_theta)
 
    f, axs = plt.subplots(nrows=n_views, ncols=2, figsize=(w,h)) 
                          #gridspec_kw={"height_ratios":[0.49, 0.49, 0.02]})
    
    for view_idx, (t_th, f_th, t_ld, f_ld, rbids) in enumerate(zip(true_theta, fitted_theta, true_logdelta, fitted_logdelta, row_batch_ids)):

        valid_idx = (rbids != "")
        t_th = t_th[valid_idx,:]
        f_th = f_th[valid_idx,:]
        t_ld = t_ld[valid_idx,:]
        f_ld = f_ld[valid_idx,:]

        # Plot theta for this view
        #t_th = t_th.flatten()
        #f_th = f_th.flatten()
        th_ax = axs[view_idx][0]
        th_ax.plot([0,0],[-100,100], "--", color="silver", linewidth=0.75)
        th_ax.plot([-100,100],[0,0], "--", color="silver", linewidth=0.75)
        #th_ax.scatter(t_th, f_th, color="k", s=0.5)
        scatter_values(th_ax, t_th, f_th)
        th_ax.set_ylabel(f"Fitted value\n\n{NAMES[batch_view_names[view_idx]]}")

        #min_v = min(np.quantile(t_th,0.001), np.quantile(f_th,0.001))
        #max_v = max(np.quantile(t_th,0.999), np.quantile(f_th,0.999))
        #th_ax.set_xlim([min_v, max_v])
        #th_ax.set_ylim([min_v, max_v])
        th_ax.set_xlim([-1, 1])
        th_ax.set_ylim([-1, 1])

        # Plot logdelta for this view
        #t_ld = t_ld.flatten()
        #f_ld = f_ld.flatten()
        ld_ax = axs[view_idx][1]
        ld_ax.plot([0,0],[-100,100], "--", color="silver", linewidth=0.75)
        ld_ax.plot([-100,100],[0,0], "--", color="silver", linewidth=0.75)
        #ld_ax.scatter(t_ld, f_ld, color="k", s=0.5)
        scatter_values(ld_ax, t_ld, f_ld)
        
        #min_v = min(np.quantile(t_ld,0.01), np.quantile(f_ld,0.01))
        #max_v = max(np.quantile(t_ld,0.99), np.quantile(f_ld,0.99))
        #ld_ax.set_xlim([min_v, max_v])
        #ld_ax.set_ylim([min_v, max_v])
        ld_ax.set_xlim([-1, 1])
        ld_ax.set_ylim([-1, 1])

    axs[0][0].set_title("Batch shift ($\\theta$)")
    axs[0][1].set_title("Batch scale (log $\\delta$)")

    axs[-1][0].set_xlabel("True value")
    axs[-1][1].set_xlabel("True value")
    plt.suptitle("Batch effect estimation")

    f.tight_layout()
    return f


def plot_param(true_hdf, fitted_hdf, target_param="batch_params", batch_view_names="methylation:mrnaseq"):

    fig = None
    with h5py.File(true_hdf, "r") as true_file:
        with h5py.File(fitted_hdf, "r") as fitted_file:

            if target_param == "batch_params":
                fig = compare_batch_params(true_file, fitted_file, batch_view_names)
            else:
                print("No method to plot comparison for ", target_param)

    return fig



if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("true_hdf")
    parser.add_argument("fitted_hdf")
    parser.add_argument("out_png")
    parser.add_argument("--param", choices=["batch_params"], default="batch_params")
    parser.add_argument("--batch_view_names", default="methylation:mrnaseq")
    parser.add_argument("--dpi", type=int, default=300)

    args = parser.parse_args()

    true_hdf = args.true_hdf
    fitted_hdf = args.fitted_hdf
    out_png = args.out_png
    target_param = args.param
    batch_view_names = args.batch_view_names.split(":")
    dpi = args.dpi

    f = plot_param(true_hdf, fitted_hdf, target_param=target_param, batch_view_names=batch_view_names)

    f.savefig(out_png, dpi=dpi)

