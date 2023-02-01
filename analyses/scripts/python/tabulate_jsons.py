
import script_util as su
import pandas as pd
import argparse
import json
import sys


def tabulate_jsons(filepaths):

    path_kvs = [su.parse_path_kvs(pth) for pth in filepaths]
    file_data = [json.load(open(pth, "r")) for pth in filepaths]

    for p_kv, f_kv in zip(path_kvs, file_data):
        p_kv.update(f_kv)

    df = pd.DataFrame(path_kvs)
    
    return df


if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("out_tsv")
    parser.add_argument("score_jsons", nargs="+")

    args = parser.parse_args()
    out_tsv = args.out_tsv
    score_jsons = args.score_jsons

    df = tabulate_jsons(score_jsons)
    df.to_csv(out_tsv, sep="\t", index=False)

