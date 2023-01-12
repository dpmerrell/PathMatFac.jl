
import script_util as su
import pandas as pd
import json
import sys


def tabulate_jsons(infer_paths, impute_paths):

    path_kvs = [su.parse_path_kvs(pth) for pth in infer_paths]

    infer_data = [json.load(open(pth, "r")) for pth in infer_paths]
    impute_data = [json.load(open(pth, "r")) for pth in impute_paths]

    for kvd, infer_d, impute_d in zip(path_kvs, infer_data, impute_data):
        kvd.update(infer_d)
        kvd.update(impute_d)

    df = pd.DataFrame(path_kvs)
    
    return df


if __name__=="__main__":
    
    args = sys.argv
    output_tsv = args[1]
    input_jsons = args[2:]

    n_jsons = len(input_jsons)
    n_settings = n_jsons//2

    infer_jsons = sorted(input_jsons[:n_settings])
    impute_jsons = sorted(input_jsons[-n_settings:])

    df = tabulate_jsons(infer_jsons, impute_jsons)
    print(df)
    df.to_csv(output_tsv, sep="\t", index=False)

