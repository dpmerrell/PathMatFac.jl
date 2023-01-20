
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import script_util as su
import pickle as pkl
import numpy as np
import argparse
import json
import h5py


def load_data(test_hdf):

    # Load the features and labels from the HDF file
    X = su.load_hdf(test_hdf, "X").transpose()
    labels = su.load_hdf(test_hdf, "target", dtype=str)

    # Encode the labels as integers.
    # (label --> its rank in sorted order)
    unq_labels = np.sort(np.unique(labels))
    encoder = {lab:i for i, lab in enumerate(unq_labels)}
    y = np.vectorize(lambda x: encoder[x])(labels)

    return X, y


def compute_scores(y, pred_y, pred_y_probs):

    scores = {'f1_micro': f1_score(y, pred_y, average="micro"),
              'f1_macro': f1_score(y, pred_y, average="macro"),
              'accuracy': accuracy_score(y, pred_y)
             }
   
    # Distinguish between multiclass and binary
    if pred_y_probs.shape[1] > 2:
        scores['roc_auc_ovr'] = roc_auc_score(y, pred_y_probs, multi_class="ovr")
        scores['roc_auc_ovo'] = roc_auc_score(y, pred_y_probs, multi_class="ovo")
    else:
        scores['roc_auc'] = roc_auc_score(y, pred_y_probs[:,1])

    return scores


if __name__=="__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("model_pkl")
    parser.add_argument("data_hdf")
    parser.add_argument("score_json")

    args = parser.parse_args()

    model_pkl = args.model_pkl
    data_hdf = args.data_hdf
    score_json = args.score_json

    # Load model
    model = pkl.load(open(model_pkl, "rb"))
 
    # Load data
    X, y = load_data(data_hdf)

    # Make predictions
    y_pred_probs = model.predict_proba(X)
    y_pred = model.predict(X)

    # Score predictions
    score_dict = compute_scores(y, y_pred, y_pred_probs)

    # Output to JSON
    json.dump(score_dict, open(score_json, "w")) 


