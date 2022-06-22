
import sys
from collections import Counter, defaultdict
import pandas as pd
import json

# Get the number of rows in the 
# "edge" block of the Reactome txt
def get_n_rows(input_txt):

    f_in = open(input_txt, "r")

    nrows=0
    for line in f_in.readlines():
        if line.isspace():
            break
        nrows += 1
    return nrows


def explore_txt(reactome_txt):

    print("\nTXT CONTENTS")
    txt_df = pd.read_csv(reactome_txt, sep="\t")

    print(txt_df.shape)
    print(txt_df.columns)

    interaction_counts = Counter(txt_df["INTERACTION_TYPE"].values)
    srt_counts = interaction_counts.most_common()
    N = txt_df.shape[0]
    for k,v in srt_counts:
        print(k, "\t\t", v/N)

    print(txt_df[txt_df["INTERACTION_TYPE"] == "ProteinReference"])

    txt_df = txt_df[pd.isnull(txt_df["PATHWAY_NAMES"]) == False]
    pwys = [name for name_str in txt_df["PATHWAY_NAMES"].values for name in name_str.split(";")]

    pwy_counts = Counter(pwys)
    srt_pwy_counts = pwy_counts.most_common()
    print("UNIQUE_PATHWAYS: ", len(srt_pwy_counts))
    for i, (k,v) in enumerate(srt_pwy_counts):
        print(i, " ", k, "\t\t", v)


def translate_interaction(itype):
    if itype == "controls-expression-of":
        return "t>"
    else:
        return "a>"


def isolate_pathways(txt_df):

    pwys = defaultdict(set) 
    
    i = 1
    for _, row in txt_df.iterrows():
       
        if i % 1000 == 0:
            print(i)

        pwy_names_str = row["PATHWAY_NAMES"]
        
        if pd.isnull(pwy_names_str) == False:
            
            pwy_names = pwy_names_str.split(";")

            interaction = translate_interaction(row["INTERACTION_TYPE"])

            for name in pwy_names:
                pwys[name].add((row["PARTICIPANT_A"], interaction, row["PARTICIPANT_B"]))

        i += 1

    names = list(pwys.keys())
    edgelists = [list(pwys[k]) for k in names]

    result = {"names": names, "pathways": edgelists}

    return result
            

def read_tsv(reactome_tsv, nrows):
    tsv_df = pd.read_csv(reactome_tsv, sep="\t", nrows=nrows)
    return tsv_df



def main():

    reactome_txt = sys.argv[1]
    output_json = sys.argv[2]

    n_rows = get_n_rows(reactome_txt)
    edge_df = read_tsv(reactome_txt, n_rows)
    pwy_dict = isolate_pathways(edge_df)

    print("Saving to JSON: ", output_json)
    with open(output_json, "w") as f:
        json.dump(pwy_dict, f) 



if __name__=="__main__":

    main()


