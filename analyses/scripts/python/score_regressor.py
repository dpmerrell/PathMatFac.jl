
from sklearn.metrics import mean_squared_error, r2_score 
import script_util as su
import pickle as pkl
import numpy as np
import argparse
import json
import h5py


def load_data(test_hdf):

    # Load the features and labels from the HDF file
    X = su.load_hdf(test_hdf, "X").transpose()
    y = su.load_hdf(test_hdf, "target", dtype=str)

    return X, y


def compute_scores(y, pred_y):

    scores = {'mse': mean_squared_error(y, pred_y),
              'r2': r2_score(y, y_pred)
             }
   
    return scores


def compute_other_attributes(y, pred_y):

    result = {"y_true": y.astype(float).tolist(),
              "y_pred": pred_y.astype(float).tolist()
             }

    return result


if __name__=="__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("model_pkl")
    parser.add_argument("data_hdf")
    parser.add_argument("score_json")
    parser.add_argument("other_output_json")
    parser.add_argument("--target", default="pathologic_stage")

    args = parser.parse_args()

    model_pkl = args.model_pkl
    data_hdf = args.data_hdf
    score_json = args.score_json
    other_output_json = args.other_output_json
    target = args.target

    # Load model
    model = pkl.load(open(model_pkl, "rb"))
 
    # Load data
    X, y = load_data(data_hdf)
    if target == "pathologic_stage":
        y = su.encode_pathologic_stage(y)

    # Make predictions
    y_pred = model.predict(X)

    # Score predictions
    score_dict = compute_scores(y, y_pred)

    # Output to JSON
    json.dump(score_dict, open(score_json, "w")) 

    # Compute other attributes of the prediction task
    other_output = compute_other_attributes(y, y_pred)
    json.dump(other_output, open(other_output_json, "w")) 
    
