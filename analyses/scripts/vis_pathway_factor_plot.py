
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np
import json
import h5py
import sys
import os

import script_util as su


def factor_lineplot(Y, features, pwy_names):
    
    df = pd.DataFrame(data=Y)
    df.columns = pwy_names

    feature_genes = [feat.split("_")[0] for feat in features]    

    df["Gene_Assay"] = features


    means = df[pwy_names].mean()
    negate_cols = means[means < 0.0].index
    df.loc[:,negate_cols] = df.loc[:,negate_cols]*(-1.0)

    fig = px.line(df, y=pwy_names,
                  hover_data=["Gene_Assay"],
                  labels={"index": "Gene_Assay",
                          "variable": "Pathway",
                          "value": "Value"
                         },
                 )
    print(fig)
 
    return fig, df 


def pathway_scatter(fig, df, pathways, pathway_names, all_genes):

    pwy_genesets = [su.pwy_to_geneset(pwy, all_genes) for pwy in pwys] 

    full_x = np.arange(Y.shape[0],dtype=int) 

    for i,gs in enumerate(pwy_genesets):

        valid_idx = [j for j,feat in enumerate(df["Gene_Assay"]) if feat.split("_")[0] in gs]
        valid_x = full_x[valid_idx]
        valid_y = df.loc[valid_idx, pathway_names[i]].values
        valid_features = df.loc[valid_idx, "Gene_Assay"].values

        fig.add_trace(go.Scatter(name=pathway_names[i],
                                 customdata=valid_features,
                                 hovertemplate="Member: %{customdata}<br></br>Value: %{y}",
                                 x=valid_x,
                                 y=valid_y,
                                 mode="markers",
                                 marker={"color":"Black", "size": 20}
                                )
                      )

    return
    


def add_pwy_menu(fig, factor_df, pwy_names):

    flags = np.identity(len(pwy_names), dtype=bool)
    flags = np.concatenate((flags,flags))

    n_pwy = len(pwy_names)

    buttons = [{"args": [{"visible": flags[:,i]}], 
                "label": pwy,
                "method": "restyle"} for i,pwy in enumerate(pwy_names)]

    all_flag = np.concatenate((np.ones(n_pwy,dtype=bool),
                               np.zeros(n_pwy,dtype=bool)
                             ))
    buttons = [{"args": [{"visible": all_flag}], 
                          "label": "All", 
                          "method": "restyle"}
               ] + buttons

    fig.update_layout(updatemenus=[{"buttons": buttons,
                                    "direction": "down",
                                    "pad":{"r": 0, "t": 0},
                                    "showactive":True,
                                    "y": 1.08,
                                    "yanchor":"top",
                                    "xanchor": "left",
                                    "x":0.1 
                                   }
                                  ]
                     )

    print(fig)
    for i, dat in enumerate(fig.data): 
        print(i, " ", type(dat))
    
    return



if __name__=="__main__":

    args = sys.argv
    model_hdf = args[1]
    pwy_json = args[2]
    out_html = args[3]
    top_k = int(args[4])

    features = su.load_features(model_hdf)
    all_genes = set([feat.split("_")[0] for feat in features])

    Y = su.load_feature_factors(model_hdf)
    Y = Y[:,:top_k]

    with open(pwy_json, "r") as f: 
        pwy_dict = json.load(f)

    pwys = pwy_dict["pathways"][:top_k]
    pwy_names = pwy_dict["names"][:top_k]
   
    fig, factor_df = factor_lineplot(Y, features, pwy_names)
    pathway_scatter(fig, factor_df, pwys, pwy_names, all_genes) 
 
    add_pwy_menu(fig, factor_df, pwy_names)

    fig.write_html(out_html)
 
