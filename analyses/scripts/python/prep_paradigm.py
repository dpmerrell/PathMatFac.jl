
import script_util as su
import pandas as pd
import numpy as np
import argparse
import h5py
import sys


def move_to_hdf(in_txt, out_hdf):

    in_df = pd.read_csv(in_txt, sep="\t", index_col=0)
    instances = in_df.columns
    features = in_df.index

    with h5py.File(out_hdf, "w") as f:
        su.write_hdf(f, "instances", instances, is_string=True)
        su.write_hdf(f, "features", features, is_string=True)
        su.write_hdf(f, "data", in_df.values.astype(float))
       
    return    


if __name__=="__main__":

    args = sys.argv
    paradigm_txt = args[1]
    out_hdf = args[2]

    move_to_hdf(paradigm_txt, out_hdf)


