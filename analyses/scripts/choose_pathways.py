
import argparse
import json
import sys
import os


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("output_json")
    parser.add_argument("--all-pathways", nargs="+")
    parser.add_argument("--dropped-pathways", nargs="*")

    args = parser.parse_args()

    dropped = set(args.dropped_pathways)
    result = [fname for fname in args.all_pathways if fname.split("/")[-1].split(".")[0] not in dropped]

    with open(args.output_json, "w") as f:
        json.dump(result, f)

