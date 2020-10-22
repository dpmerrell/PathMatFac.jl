
import pandas as pd
import json
import argparse


def create_tcga_array(args):

    with open(args.tcga_pwy_json) as f:
        pwy_dat = json.load(f)
        pwy_entities = pwy_dat["entity_names"]

    cnv_df = pd.read_csv(args.cnv_file, index_col=0)
    cnv_df.columns = [col + "::DNA" for col in cnv_df.columns]

    gene_expr_df = pd.read_csv(args.gene_expr_file, index_col=0)
    gene_expr_df.columns = [col + "::MRNA" for col in gene_expr_df.columns]

    results = pd.DataFrame(columns=pwy_entities, index=cnv_df.index)
    results.loc[:,gene_expr_df.columns] = gene_expr_df
    results.loc[:,cnv_df.columns] = cnv_df

    print(results)

    return results.values.tolist()
    

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("tcga_pwy_json")
    parser.add_argument("cnv_file")
    parser.add_argument("gene_expr_file")
    parser.add_argument("output_file")
    args = parser.parse_args()

    arr = create_tcga_array(args)

    result = { "data": arr }

    with open(args.output_file, "w") as f:
        json.dump(result, f)

