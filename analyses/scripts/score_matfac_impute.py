
import script_util as su
import numpy as np
import json
import sys

def score_imputation(imputed_hdf, masked_hdf, true_hdf):

    masked_data = su.load_omic_data(masked_hdf)
    ismasked = np.isnan(masked_data)
    masked_data = None
    
    #imputed_data = su.load_omic_data(imputed_hdf)
    #true_data = su.load_omic_data
    return score_dict


if __name__=="__main__":

    args = sys.argv
    imputed_hdf = args[1]
    masked_hdf = args[2]
    true_hdf = args[3]
    score_json = args[4]

    score_dict = score_imputation(imputed_hdf, masked_hdf, true_hdf) 
    json.dump(score_dict, open(score_json, "w"))


