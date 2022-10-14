
import script_util as su
import pandas as pd
import json
import sys


def tabulate_jsons(json_paths):

    path_kvs = [su.parse_path_kvs(pth) for pth in json_paths]

    json_data = [json.load(open(pth, "r")) for pth in json_paths]

    for kvd, jsd in zip(path_kvs, json_data):
        jsd.update(kvd)

    df = pd.DataFrame(json_data)
    return df


if __name__=="__main__":
    
    args = sys.argv
    output_tsv = args[1]
    input_jsons = args[2:]

    df = tabulate_jsons(input_jsons)
    print(df)
    df.to_csv(output_tsv, sep="\t", index=False)

