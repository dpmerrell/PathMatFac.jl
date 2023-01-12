
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score
import script_util as su
import numpy as np
import json
import sys


assay_to_score = {"mrnaseq": r2_score,
                  "cna": r2_score,
                  "mutation": average_precision_score,
                  "methylation": r2_score,
                  "rppa": r2_score
                 }

def score_imputation(imputed_hdf, masked_hdf, true_hdf):

    # Load the masked data; find the indices of masked entries
    masked_data = su.load_hdf(masked_hdf, "omic_data/data")
    ismasked = np.isnan(masked_data)
    masked_data = None
    
    # Load the feature assays
    feature_assays = su.load_hdf(imputed_hdf, 
                                 "omic_data/feature_assays", 
                                 dtype=str)

    # Load the true and imputed data
    imputed_data = su.load_hdf(imputed_hdf, "omic_data/data")
    true_data = su.load_hdf(true_hdf, "omic_data/data")

    # For each assay, compute a score
    result = {}
    unq_assays = np.unique(feature_assays)
    for assay in unq_assays:
        relevant_features = (feature_assays == assay)
        relevant_mask = ismasked[relevant_features,:]
        relevant_pred = imputed_data[relevant_features,:]
        relevant_true = true_data[relevant_features,:]
        
        score = 1.0 
        if np.sum(relevant_mask) > 0: 
            masked_pred = relevant_pred[relevant_mask]
            masked_true = relevant_true[relevant_mask]
            score = assay_to_score[assay](masked_true, masked_pred)

        result["{}_impute_score".format(assay)] = score 
    return result


if __name__=="__main__":

    args = sys.argv
    imputed_hdf = args[1]
    masked_hdf = args[2]
    true_hdf = args[3]
    score_json = args[4]

    score_dict = score_imputation(imputed_hdf, masked_hdf, true_hdf) 
    json.dump(score_dict, open(score_json, "w"))


