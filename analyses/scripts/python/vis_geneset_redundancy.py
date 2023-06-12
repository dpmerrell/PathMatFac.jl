
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
import script_util as su
import vis_model_params as vmp
import numpy as np
import argparse

mpl.rcParams['font.family'] = "serif"
#plt.rcParams.update({'text.usetex': True,
#                     'font.family': 'serif'})

NAMES = su.NICE_NAMES


def pwy_to_geneset(pwy):
    return set([edge[0] for edge in pwy])

def pwy_to_geneset(pwy):
    return

def plot_gene_sets(in_json, plot="matrix"):

    fig = None
    with open(in_json, "r") as in_f:
        json_contents = json.load(in_f)
        pwys = json_contents["pathways"]
        names = json_contents["names"]

    genesets = [pwy_to_geneset(pwy) for pwy in pwys]
    all_genes = set()
    for gs in genesets:
        all_genes |= gs


    if target_param == "matrix":
        pass 
    elif target_param == "heatmap":
        pass

    return fig


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("pwys_json")
    parser.add_argument("out_png")
    parser.add_argument("--plot", choices=["matrix", "heatmap"], default="matrix")
    parser.add_argument("--dpi", type=int, default=300)

    args = parser.parse_args()

    in_json = args.pwys_json
    out_png = args.out_png
    target_plot = args.plot
    dpi = args.dpi

    f = plot_gene_sets(in_json, target_plot=target_plot)
    f.savefig(out_png, dpi=dpi)


