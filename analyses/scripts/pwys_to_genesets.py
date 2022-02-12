
import json
import argparse
import script_util as su



if __name__=="__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("input_json")
    parser.add_argument("output_json")

    args = parser.parse_args()

    with open(args.input_json, "r") as f:
        pwy_dict = json.load(f)
        print("PWY_DICT: ", pwy_dict)

    genesets = [su.pwy_to_geneset(pwy) for pwy in pwy_dict["pathways"]]
    
    result = { "genesets": genesets,
               "names": pwy_dict["names"]
             }

    with open(args.output_json, "w") as f:
        json.dump(result, f)
   
