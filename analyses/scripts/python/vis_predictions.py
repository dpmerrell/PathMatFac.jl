
from collections import defaultdict
from matplotlib import pyplot as plt
import matplotlib as mpl
import script_util as su
from os import path
import pandas as pd
import numpy as np
import argparse


NAMES = su.NICE_NAMES
TASKS = su.ALL_TASKS
METHODS = su.ALL_METHODS

def sort_by_order(ls, ordered_vocab):
    ordering_dict = {x:i for i,x in enumerate(ordered_vocab)}
    srt_idx = sorted([ordering_dict[x] for x in ls])
    return [ordered_vocab[i] for i in srt_idx]


def get_method_target(result_json):
    path_kvs = su.parse_path_kvs(result_json)
    method = path_kvs["method"]
    target = path_kvs["target"]
    return method, target


def get_methods_targets(result_jsons):
    """
    Given a list of JSON filepaths, store them in 
    a nested dictionary indexed by "method" and "target".
    """
    result = defaultdict(lambda : defaultdict(list)) 
    for rjs in result_jsons:
        method, target = get_method_target(rjs)
        result[method][target].append(rjs)

    return result
    

def dict_to_grid(d, row_order=METHODS, col_order=TASKS):

    rownames = list(d.keys())
    colnames = set()
    for rname in rownames:
        colnames |= d[rname]
    
    rownames = sort_by_order(rownames, row_order)
    colnames = sort_by_order(list(colnames), col_order)
    
    mat = [[d[rname][cname] for cname in colnames] for rname in rownames]

    return mat, rownames, colnames



if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("out_png")
    parser.add_argument("--result_jsons", nargs="+")

    args = parser.parse_args()
    result_jsons = args.result_jsons
    out_png = args.out_png

    method_target_jsons = get_methods_targets(result_jsons)
    

