
import chart_studio.plotly as cs
import plotly.express as px

import pandas as pd
import numpy as np
import json
import h5py
import sys
import os

import script_util as su


def plot_factors(Y, feature_genes, feature_assays,
                    pwy_genesets, pwy_names):
    
    df = pd.DataFrame(data=Y.transpose())
    df.columns = pwy_names
    
    df["Gene"] = feature_genes
    df["Assay"] = feature_assays

    means = df[pwy_names].mean()
    negate_cols = means[means < 0.0].index
    df.loc[:,negate_cols] = df.loc[:,negate_cols]*(-1.0)

    fig = px.line(df, y=pwy_names,
                      hover_data=["Gene", "Assay"],
                      labels={"index": "(Assay, Gene)",
                              "variable": "Pathway",
                              "value": "Value"
                             },
                      title="Pathway Factors"
                 )
 
    return fig 


if __name__=="__main__":

    args = sys.argv
    model_hdf = args[1]
    pwy_json = args[2]
    out_html = args[3]

    with h5py.File(model_hdf, "r") as f:
        Y = f["matfac"]["Y"][:,:].transpose()
        feature_genes = f["augmented_genes"][:].astype(str)
        feature_assays = f["augmented_assays"][:].astype(str)

    orig_idx = (feature_assays != "")
    Y = Y[:,orig_idx]
    feature_genes = feature_genes[orig_idx]
    feature_assays = feature_assays[orig_idx]

    with open(pwy_json, "r") as f: 
        pwy_dict = json.load(f)

    pwys = pwy_dict["pathways"]
    pwy_names = pwy_dict["names"]
    pwy_genesets = [su.pwy_to_geneset(pwy) for pwy in pwys] 
   
    fig = plot_factors(Y, feature_genes, feature_assays, 
                          pwy_genesets, pwy_names)

    fig.write_html(out_html)
 
