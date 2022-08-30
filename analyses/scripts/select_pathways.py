
import json
import sys

if __name__=="__main__":

    args = sys.argv
    ranked_json = args[1]
    topk = int(args[2])
    out_json = args[3]

    ranked_pwy_dict = json.load(open(ranked_json, "r"))

    selected = {"names": ranked_pwy_dict["names"][:topk],
                "pathways": ranked_pwy_dict["pathways"][:topk]}

    json.dump(selected, open(out_json, "w"))


